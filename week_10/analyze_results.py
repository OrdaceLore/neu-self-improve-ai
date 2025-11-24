import json

def analyze():
    with open("experiment_results.json", "r") as f:
        results = json.load(f)

    successes = [r for r in results if r['correct']]
    failures = [r for r in results if not r['correct']]

    print(f"Total analyzed: {len(results)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    print("-" * 40)

    # --- LOGIC FOR Q3: 5 Easiest Failures ---
    # Metric: "Easiest" usually means short question length or fewer inference steps required.
    # We will sort by length of the question text (shortest = easiest).
    failures.sort(key=lambda x: len(x['question']))
    
    print("### Q3: 5 Easiest Examples where MaAS FAILED ###")
    for i, item in enumerate(failures[:5]):
        print(f"{i+1}. ID: {item['id']}")
        print(f"   Question Length: {len(item['question'])} chars")
        print(f"   Question: {item['question'].split('Options:')[0].strip()}...")
        print(f"   Expected: {item['ground_truth']} | Got: {item['model_prediction']}")
        print("")

    # --- LOGIC FOR Q4: 5 Hardest Successes ---
    # Metric: "Hardest" usually means high complexity. 
    # We use 'steps_taken' or agent graph depth as a proxy.
    successes.sort(key=lambda x: x.get('steps_taken', 0), reverse=True)

    print("### Q4: 5 Hardest Examples where MaAS SUCCEEDED ###")
    for i, item in enumerate(successes[:5]):
        print(f"{i+1}. ID: {item['id']}")
        print(f"   Complexity (Steps/Tokens): {item.get('steps_taken', 0)}")
        print(f"   Question: {item['question'].split('Options:')[0].strip()}...")
        print(f"   Agent Graph: {item.get('agent_graph', 'N/A')}")
        print("")

if __name__ == "__main__":
    analyze()