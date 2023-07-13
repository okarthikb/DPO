import torch
import os
import wandb
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import autocast, GradScaler
from transformers import PretrainedConfig, GPT2LMHeadModel
from typing import List, Tuple, Dict
from argparse import ArgumentParser


def initialize_logger(project, run):
  wandb.init(project=project)
  wandb.run.name = run


def log(loss, step, interval):
  if step % interval == 0:
    # wandb.log({'loss': loss}, step=step)
    print(f'step = {step}\tloss = {loss}')


def pad_tensor(seq, max_len, pad_value):
  pad_len = max_len - seq.shape[0]
  if pad_len <= 0:
    return seq
  return torch.cat([seq, torch.ones(pad_len, dtype=torch.long) * pad_value])


def get_max_len(examples):
  chosen, rejected = 'prompt_chosen_tokens', 'prompt_rejected_tokens'
  return max(
    max(len(example[chosen]), len(example[rejected])) for example in examples
  )


def get_padded_batch(examples: List[Dict], max_len: int, device: str) -> Tuple[torch.Tensor]:
  key_pct, key_prt = 'prompt_chosen_tokens', 'prompt_rejected_tokens'
  key_clm, key_rlm = 'chosen_loss_mask', 'rejected_loss_mask'
  chosen_tokens = torch.stack([
    pad_tensor(example[key_pct], max_len, 1) for example in examples
  ]).to(device)
  rejected_tokens = torch.stack([
    pad_tensor(example[key_prt], max_len, 1) for example in examples
  ]).to(device)
  chosen_loss_masks = torch.stack([
    pad_tensor(example[key_clm], max_len, 0) for example in examples
  ]).to(device)
  rejected_loss_masks = torch.stack([
    pad_tensor(example[key_rlm], max_len, 0) for example in examples
  ]).to(device)
  return chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks


def get_log_ps(logits, idxs, loss_mask):
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
  policy_log_ratio = chosen_policy_log_ps - rejected_policy_log_ps
  ref_log_ratio = chosen_ref_log_ps - rejected_ref_log_ps
  loss = -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio)).mean()
  return loss


def compute_loss(policy_model, ref_model, batch):
  chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks = batch

  with autocast():
    chosen_policy_logits = policy_model(chosen_tokens).logits
    rejected_policy_logits = policy_model(rejected_tokens).logits
    chosen_policy_log_ps = get_log_ps(
      chosen_policy_logits, chosen_tokens, chosen_loss_masks
    )
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
      beta=0.01
    )


def process(gpu, args):
  world_size = args.nodes * args.gpus
  rank = args.node * args.gpus + gpu

  dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

  print(f'Started process {rank}')

  device = torch.device(f'cuda:{rank}')

  n_positions = PretrainedConfig.from_pretrained('gpt2').n_positions

  policy_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
  policy_model = FSDP(
    policy_model,
    process_group=dist.new_group([i for i in range(world_size)]),
    device_id=rank
  )
  policy_model.train()
  optimizer = Adam(policy_model.parameters(), lr=args.lr)

  ref_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
  ref_model.eval()

  dataset = torch.load('dataset.pt')

  # if rank == 0:
  #   initialize_logger(args.project, args.run)

  scaler = GradScaler()
  for step in range(1, args.steps + 1):
    examples = random.sample(dataset, args.batch_size)
    max_len = min(n_positions, get_max_len(examples))
    batch = get_padded_batch(examples, max_len, device=device)
    loss = compute_loss(policy_model, ref_model, batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if rank == 0:
      log(loss.item(), step, args.log_interval)
 
  # if rank == 0:
    # torch.save(policy_model.state_dict(), 'policy_model.pt')

  dist.destroy_process_group()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--nodes', type=int, default=1)
  parser.add_argument('--gpus', type=int, default=4)
  parser.add_argument('--node', type=int, default=0)
  parser.add_argument('--steps', type=int, default=1000)
  parser.add_argument('--log_interval', type=int, default=1)
  parser.add_argument('--project', type=str, default='gpt2-large_anthropic-hh-rlhf_DPO')
  parser.add_argument('--run', type=str, default='1')
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--batch_size', type=int, default=4)
  args = parser.parse_args()

  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '6969'

  mp.spawn(process, args=(args,), nprocs=args.nodes * args.gpus, join=True)
