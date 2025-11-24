import json
from datasets import load_dataset
import random

# 1. Load CommonsenseQA (Standard Leaderboard Benchmark)
# We use a small subset to satisfy the "3 min training" constraint
print("Loading CommonsenseQA dataset...")
dataset = load_dataset("commonsense_qa", split="train")

# 2. Format for MaAS
# MaAS typically expects: { "question": str, "answer": str }
formatted_data = []

for item in dataset:
    # Convert letter answer (A,B,C,D,E) to the actual text for easier agent verification
    choices = item['choices']
    answer_key = item['answerKey']
    
    # Find the text corresponding to the answer key
    answer_text = ""
    options_text = []
    for label, text in zip(choices['label'], choices['text']):
        formatted_str = f"{label}: {text}"
        options_text.append(formatted_str)
        if label == answer_key:
            answer_text = text

    full_question = f"{item['question']}\nOptions:\n" + "\n".join(options_text)
    
    formatted_data.append({
        "id": item['id'],
        "question": full_question,
        "ground_truth": answer_key, # Keeping the letter as the strict truth
        "metadata": {"difficulty": "unknown"} # Placeholder
    })

# 3. Subsampling for Speed
# To finish search in < 3 mins, use a tiny train set (e.g., 10 samples) 
# and a small test set (e.g., 20 samples).
train_set = formatted_data[:10]
test_set = formatted_data[10:60] # 50 samples for testing

print(f"Prepared {len(train_set)} training items and {len(test_set)} testing items.")

# 4. Save to JSONL (Standard format for Agent repos)
def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

save_jsonl(train_set, "benchmark_train.jsonl")
save_jsonl(test_set, "benchmark_test.jsonl")
print("Files saved: benchmark_train.jsonl, benchmark_test.jsonl")