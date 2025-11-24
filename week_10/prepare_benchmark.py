import json
from datasets import load_dataset

# 1. Load BoolQ (Boolean Questions) - Standard Leaderboard Benchmark
# We use a small subset to satisfy the "3 min training" constraint
# BoolQ tests reading comprehension with yes/no questions
print("Loading BoolQ dataset...")
dataset = load_dataset("boolq", split="train")

# 2. Format for MaAS
# MaAS typically expects: { "question": str, "answer": str }
formatted_data = []

for item in dataset:
    # BoolQ has a question, passage, and answer (True/False)
    question = item['question']
    passage = item['passage']
    answer = item['answer']  # True or False
    
    # Format as multiple choice with passage context
    full_question = f"Passage: {passage}\n\nQuestion: {question}\nOptions:\nA: Yes\nB: No"
    answer_key = "A" if answer else "B"
    
    formatted_data.append({
        "id": item.get('idx', len(formatted_data)),  # Use idx if available, otherwise index
        "question": full_question,
        "ground_truth": answer_key, # A or B
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