import os, random, torch, wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam

from torch.distributed.fsdp import (
  FullyShardedDataParallel as FSDP,
  CPUOffload,
  MixedPrecision
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from transformers import AutoModelForCausalLM
from argparse import ArgumentParser
from functools import partial


KEY_PCT = 'prompt_chosen_tokens'
KEY_PRT = 'prompt_rejected_tokens'
KEY_CLM = 'chosen_loss_mask'
KEY_RLM = 'rejected_loss_mask'


def initialize_logger(project, run):
  wandb.init(project=project)
  wandb.run.name = run


def log(loss, chosen_reward, rejected_reward, step, interval, flag):
  if step % interval == 0:
    print(
      f'step = {step}\tloss = {loss}\tchosen_reward = {chosen_reward}\trejected_reward = {rejected_reward}'
    )
    if flag:
      wandb.log(
        {'loss': loss, 'chosen_reward': chosen_reward, 'rejected_reward': rejected_reward},
        step=step
      )


def pad_tensor(seq, max_len, pad_value):
  """
  args:
    seq: A tensor of shape (seq_len,)
    max_len: The length to pad to
    pad_value: The value to pad with

  returns:
    A tensor of shape (max_len,)
  """

  pad_len = max_len - seq.shape[0]
  if pad_len <= 0:
    return seq[:max_len]
  return torch.cat([seq, torch.ones(pad_len, dtype=torch.long) * pad_value])


def get_max_len(examples):
  """
  args:
    examples:
      A list of examples, where each example is a dict with chosen and rejected
      input tensors (along with loss masks)
  
  returns:
    The length of the longest chosen or rejected input tensor
  """

  return max(
    max(len(example[KEY_PCT]), len(example[KEY_PRT])) for example in examples
  )


def get_padded_batch(examples, max_len, device='cpu'):
  """
  args:
    examples: A list of examples, each a dict with 4 key-value pairs
    max_len: The length each input tensor should be padded to
    device: The device to put the tensors on, default is 'cpu'
  
  returns:
    A tuple of 4 tensors, each of shape (batch_size, max_len)
  """
  
  chosen_tokens = torch.stack([
    pad_tensor(example[KEY_PCT], max_len, 1) for example in examples
  ]).to(device)

  rejected_tokens = torch.stack([
    pad_tensor(example[KEY_PRT], max_len, 1) for example in examples
  ]).to(device)

  chosen_loss_masks = torch.stack([
    pad_tensor(example[KEY_CLM], max_len, 0) for example in examples
  ]).to(device)

  rejected_loss_masks = torch.stack([
    pad_tensor(example[KEY_RLM], max_len, 0) for example in examples
  ]).to(device)

  return chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks


def get_log_ps(logits, idxs, loss_mask):
  """
  args:
    logits: A tensor of shape (batch_size, seq_len, vocab_size)
    idxs: A torch.long tensor of shape (batch_size, seq_len)
    loss_mask: A torch.float tensor of shape (batch_size, seq_len)
  
  returns:
    A tensor of shape (batch_size,), the log probabilities of each sequence in the batch
  """

  idxs = idxs[:, 1:].unsqueeze(2)
  loss_mask = loss_mask[:, 1:]
  log_p_distributions = F.log_softmax(logits, dim=-1)[:, :-1]
  log_ps = torch.gather(log_p_distributions, dim=2, index=idxs).squeeze(2)
  return (log_ps * loss_mask).sum(dim=-1)


def loss_fn(
  chosen_policy_log_ps,
  rejected_policy_log_ps,
  chosen_ref_log_ps,
  rejected_ref_log_ps,
  beta=0.01
):
  """
  args:
    chosen_policy_log_ps: A tensor of shape (batch_size,)
    rejected_policy_log_ps: A tensor of shape (batch_size,)
    chosen_ref_log_ps: A tensor of shape (batch_size,)
    rejected_ref_log_ps: A tensor of shape (batch_size,)
    beta: The KL penalty parameter, default is 0.01 (from the paper)
  
  returns:
    A scalar tensor, the loss, and two scalar tensors, the chosen and rejected rewards
  """

  policy_log_ratio = chosen_policy_log_ps - rejected_policy_log_ps
  ref_log_ratio = chosen_ref_log_ps - rejected_ref_log_ps
  loss = -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio)).mean()
  
  # compute rewards too
  with torch.no_grad():
    chosen_reward = beta * (chosen_policy_log_ps - chosen_ref_log_ps).sum().cpu()
    rejected_reward = beta * (rejected_policy_log_ps - rejected_ref_log_ps).sum().cpu()

  return loss, chosen_reward, rejected_reward


def compute_loss(policy_model, ref_model, batch, beta):
  """
  args:
    policy_model: The policy model, $\pi_{\theta}$
    ref_model: The reference model, $\pi_{\phi}$
    batch: A tuple of 4 tensors, each of shape (batch_size, max_len)

  returns:
    A scalar tensor, the loss, and two scalar tensors, the chosen and rejected rewards
  """

  chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks = batch

  chosen_policy_logits = policy_model(chosen_tokens).logits
  chosen_policy_log_ps = get_log_ps(
    chosen_policy_logits, chosen_tokens, chosen_loss_masks
  )

  rejected_policy_logits = policy_model(rejected_tokens).logits
  rejected_policy_log_ps = get_log_ps(
    rejected_policy_logits, rejected_tokens, rejected_loss_masks
  )

  with torch.no_grad():
    chosen_ref_logits = ref_model(chosen_tokens).logits
    rejected_ref_logits = ref_model(rejected_tokens).logits
    chosen_ref_log_ps = get_log_ps(
      chosen_ref_logits, chosen_tokens, chosen_loss_masks
    )
    rejected_ref_log_ps = get_log_ps(
      rejected_ref_logits, rejected_tokens, rejected_loss_masks
    )

  return loss_fn(
    chosen_policy_log_ps,
    rejected_policy_log_ps,
    chosen_ref_log_ps,
    rejected_ref_log_ps,
    beta=beta
  )


def process(gpu, args):
  world_size = args.nodes * args.gpus
  rank = args.node * args.gpus + gpu

  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

  print(f'Started process {rank}')

  device = torch.device(f'cuda:{rank}')

  Layer = GPT2Block if 'gpt2' in args.model else GPTNeoXLayer

  auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Layer}
  )

  mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
  )

  # load and shard policy model, $\pi_{\theta}$
  policy_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
  policy_model = FSDP(
    policy_model,
    process_group=dist.new_group([i for i in range(world_size)]),
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision,
    cpu_offload=CPUOffload(offload_params=args.cpu_offload),
    device_id=torch.cuda.current_device()
  )
  # train mode
  policy_model.train()
  
  if rank == 0:
    print('Loaded and sharded policy model')

  # apply activation checkpointing to policy model
  if args.activation_checkpointing:

    # check if activation checkpointing is available
    try:
      from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
      )
      
      non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
      )
      check_fn = lambda submodule: isinstance(submodule, Layer)
      apply_activation_checkpointing(
        policy_model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
      )
      
      if rank == 0:
        print('Applied activation checkpointing')
    
    except ImportError:
      if rank == 0:
        print('Activation checkpointing not available :(')

  # load and shard reference (pretrained) model, $\pi_{\phi}$
  ref_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
  ref_model = FSDP(
    ref_model,
    process_group=dist.new_group([i for i in range(world_size)]),
    mixed_precision=mixed_precision,
    cpu_offload=CPUOffload(offload_params=args.cpu_offload),
    device_id=torch.cuda.current_device()
  )
  # eval mode, we won't be training it
  ref_model.eval()
  print('Loaded and sharded reference model')
  
  optimizer = Adam(policy_model.parameters(), lr=args.lr)
  dataset = torch.load('dataset.pt')

  if rank == 0 and args.wandb:
    assert [args.project, args.run_name] != [None, None], 'Project and run name must be specified'
    initialize_logger(args.project, args.run_name)

  # sync processes
  dist.barrier()

  # get model context length
  ctx_len = policy_model.config.max_position_embeddings

  for step in range(1, args.steps + 1):
    examples = random.sample(dataset, args.batch_size)
    max_len = min(ctx_len, get_max_len(examples))
    batch = get_padded_batch(examples, max_len, device=device)
    loss, chosen_reward, rejected_reward = compute_loss(
      policy_model, ref_model, batch, args.beta
    )
    loss.backward()
    optimizer.step() 
    optimizer.zero_grad()
    if rank == 0:
      log(
        loss.item(),
        chosen_reward.item(),
        rejected_reward.item(),
        step, 
        args.log_interval,
        args.wandb
      )
  
  dist.barrier()
 
  if rank == 0:
    policy_model.save_pretrained('policy_model.pth')

  dist.destroy_process_group()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--gpus', type=int, default=4)
  parser.add_argument('--node', type=int, default=0)
  parser.add_argument('--steps', type=int, default=10)
  parser.add_argument('--log_interval', type=int, default=1)
  parser.add_argument('--model', type=str, default='EleutherAI/Pythia-2.8B')
  parser.add_argument('--lr', type=float, default=1e-6)
  parser.add_argument('--beta', type=float, default=0.1)
  parser.add_argument('--batch_size', type=int, default=4)
  parser.add_argument('--cpu_offload', type=bool, default=False)
  parser.add_argument('--activation_checkpointing', type=bool, default=True)
  parser.add_argument('--wandb', type=bool, default=False)
  parser.add_argument('--project', type=str, default=None)
  parser.add_argument('--run_name', type=str, default=None)
  args = parser.parse_args()

  assert args.model in [
    'EleutherAI/Pythia-70M',
    'EleutherAI/Pythia-160M',
    'EleutherAI/Pythia-410M',
    'EleutherAI/Pythia-1B',
    'EleutherAI/Pythia-1.4B',
    'EleutherAI/Pythia-2.8B',
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl'
  ], 'Invalid model name'

  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '4269'

  mp.spawn(process, args=(args,), nprocs=args.nodes * args.gpus, join=True)
