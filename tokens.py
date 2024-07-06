from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/gemma-2-9b-bnb-4bit"
)

BASE_FILENAMES = [
    f"data/{{}}-1.txt",
    f"data/{{}}-2.txt",
    f"data/{{}}-3.txt",
    f"data/{{}}-4.txt",
    f"data/{{}}-5.txt",
]

for i, base_filename in enumerate(BASE_FILENAMES):
  with open(base_filename.format("story"), "r") as file:
    story = "".join(file.readlines())
  with open(base_filename.format("summary"), "r") as file:
    summary = "".join(file.readlines())
  
  story_tokens = tokenizer(story, return_tensors="pt")["input_ids"].shape[1]
  summary_tokens = tokenizer(summary, return_tensors="pt")["input_ids"].shape[1]

  print(f"Story {i+1}", story_tokens, summary_tokens, story_tokens+summary_tokens)