## DPO

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) is an alternative to RLHF that optimizes the same objective but doesn't require a reward model or online RL. It is much cleaner to implepement than say, PPO (Proximal Policy Optimization). The dataset is a `.pt` file with dicts, where each dict has keys `prompt_chosen_tokens` (tensor, prompt tokens and chosen response tokens), `prompt_rejected_tokens` (prompt tokens and rejected response tokens), `chosen_loss_mask` (the loss mask for `prompt_chosen_tokens`, we only compute loss for the response tokens), and `rejected_loss_mask` (for `prompt_rejected_tokens`).

Dataset is generated by `dataset.py` using Anthropic's HH-RLHF `jsonl` files [here](https://github.com/anthropics/hh-rlhf/tree/master/harmless-base). For non-Ampere GPUs, change `{param, reduce, buffer}_dtype` in `mixed_precision` in `train.py` to something other than `bfloat16`.

Check out my [post](http://okarthikb.github.io/site/blog/dpo.html) for a more in-depth explanation of DPO.

To train, get the `jsonl` files. Then

```
python3 dataset.py --model <path to HF model> --dataset <path to jsonl file>
python3 train.py --nodes <number of nodes> --gpus <gpus per node> --model <path to HF model> ...
```

For `<path to HF model>`, use Eleuther's Pythia or the GPT-2 models. Training is sped up with FSDP and activation recomputation.

<div align="center">
  <img src="https://github.com/okarthikb/DPO/assets/86470305/468ca087-1e00-4429-905f-55a4c3c947c1"/>
  
  <img src="https://github.com/okarthikb/DPO/assets/86470305/8711c30f-63e0-4269-9841-4030515b5a5f"/>
  
  <img src="https://github.com/okarthikb/DPO/assets/86470305/9b09332a-3f24-4613-9a2c-03a5b17bc937"/>
</div>

### Relevant reading

1. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
2. [Gradient-Checkpointing in PyTorch](https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html)
3. [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
4. [DPO](https://arxiv.org/abs/2305.18290)
5. [RLHF](https://arxiv.org/abs/1706.03741)
