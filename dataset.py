import torch, json
from transformers import AutoTokenizer
from argparse import ArgumentParser


def main(args):
  tokenizer = AutoTokenizer.from_pretrained(args.model)

  with open(args.dataset, 'r') as f:
    lines = f.readlines()

  dataset = []
  search_term = '\n\nAssistant:'

  for i, line in enumerate(lines):
    example = json.loads(line)

    try:
      chosen_idx = example['chosen'].rfind(search_term)
      rejected_idx = example['rejected'].rfind(search_term)
      assert chosen_idx != -1 and rejected_idx != -1, f'Search term not found for example {i}'

      chosen_idx = chosen_idx + len(search_term)
      rejected_idx = rejected_idx + len(search_term) 
      chosen_prompt = example['chosen'][:chosen_idx]
      rejected_prompt = example['rejected'][:rejected_idx]
      chosen_response = example['chosen'][chosen_idx:] 
      rejected_response = example['rejected'][rejected_idx:] 
      assert chosen_prompt == rejected_prompt, f'Prompts do not match, ignoring example {i}'

      prompt_tokens = tokenizer.encode(chosen_prompt, return_tensors='pt')
      chosen_response_tokens = tokenizer.encode(chosen_response, return_tensors='pt')
      rejected_response_tokens = tokenizer.encode(rejected_response, return_tensors='pt')

      prompt_chosen_tokens = torch.cat([prompt_tokens, chosen_response_tokens], dim=1)
      prompt_rejected_tokens = torch.cat([prompt_tokens, rejected_response_tokens], dim=1)

      chosen_loss_mask = torch.cat(
        [torch.zeros(prompt_tokens.shape), torch.ones(chosen_response_tokens.shape)], dim=1
      )
      rejected_loss_mask = torch.cat(
        [torch.zeros(prompt_tokens.shape), torch.ones(rejected_response_tokens.shape)], dim=1
      )

      dataset_example = {
        'prompt_chosen_tokens': prompt_chosen_tokens.squeeze(),
        'prompt_rejected_tokens': prompt_rejected_tokens.squeeze(),
        'chosen_loss_mask': chosen_loss_mask.squeeze(),
        'rejected_loss_mask': rejected_loss_mask.squeeze()
      }

      dataset.append(dataset_example)

    except Exception as e:
      print(e)

  torch.save(dataset, 'dataset.pt')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument(
    '--model',
    help="Hugging Face path to model's tokenizer",
    type=str,
    default='gpt2'
  )
  parser.add_argument(
    '--dataset',
    help='path to dataset',
    type=str,
    default='test.jsonl'
  )
  args = parser.parse_args()

  main(args)
