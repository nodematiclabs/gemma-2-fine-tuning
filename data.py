prompt = """Below is a story. Please summarize this in my style.

### Story:
{}

### Summary:
{}"""

BASE_FILENAMES = [
    f"data/{{}}-1.txt",
    f"data/{{}}-2.txt",
    f"data/{{}}-3.txt",
    f"data/{{}}-4.txt",
    f"data/{{}}-5.txt",
]

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

finetuning_examples = []
for base_filename in BASE_FILENAMES:
  with open(base_filename.format("story"), "r") as file:
    story = "".join(file.readlines())
  with open(base_filename.format("summary"), "r") as file:
    summary = "".join(file.readlines())
  finetuning_examples.append({
      "text": prompt.format(story, summary) + EOS_TOKEN
  })

from datasets import Dataset
dataset = Dataset.from_list(finetuning_examples)