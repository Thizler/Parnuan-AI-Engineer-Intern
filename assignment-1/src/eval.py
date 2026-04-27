import json
import time
import os
import statistics
from tqdm import tqdm
from ner import NERSystem
from dotenv import load_dotenv

load_dotenv()

def calculate_metrics(tp, fp, fn):
    """คำนวณ Precision, Recall และ F1 Score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def run_evaluation(model_id, dataset_path, price_per_1k):
    try:
        ner = NERSystem(model_name=model_id)
    except ValueError as e:
        print(e)
        return

    # Metrics รายฟิลด์
    metrics = {"amount": {"tp": 0, "fp": 0, "fn": 0}, "detail": {"tp": 0, "fp": 0, "fn": 0}}
    
    # สถิติระดับ Message
    exact_matches = 0
    correct_counts = 0
    
    # Failure Taxonomy
    taxonomy = {
        "missed_transaction": 0,
        "hallucinated_transaction": 0,
        "wrong_amount": 0,
        "wrong_detail": 0
    }
    
    latencies = []
    regex_hits = 0 

    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    total = len(samples)
    print(f"\n🔍 Evaluating: {model_id}")
    
    for sample in tqdm(samples):
        text = sample['text']
        gt_list = sample['transactions']
        
        # 1. Regex Hybrid Hit check
        if ner.parse_with_regex(text) is not None:
            regex_hits += 1

        # 2. Performance & Output
        start_time = time.time()
        response = ner.parse(text)
        latencies.append(time.time() - start_time)
        
        pred_list = [t.model_dump() for t in response.transactions]
        
        # 3. Exact Match & Count Accuracy
        if pred_list == gt_list:
            exact_matches += 1
        
        if len(pred_list) == len(gt_list):
            correct_counts += 1
        else:
            if len(pred_list) < len(gt_list):
                taxonomy["missed_transaction"] += (len(gt_list) - len(pred_list))
            else:
                taxonomy["hallucinated_transaction"] += (len(pred_list) - len(gt_list))

        # 4. Field-level Metrics Logic
        for i in range(max(len(gt_list), len(pred_list))):
            if i < len(gt_list) and i < len(pred_list):
                # Check Amount
                if pred_list[i]['amount'] == gt_list[i]['amount']: 
                    metrics["amount"]["tp"] += 1
                else: 
                    metrics["amount"]["fp"] += 1
                    taxonomy["wrong_amount"] += 1
                
                # Check Detail
                if str(pred_list[i]['detail']).strip() == str(gt_list[i]['detail']).strip(): 
                    metrics["detail"]["tp"] += 1
                else: 
                    metrics["detail"]["fp"] += 1
                    taxonomy["wrong_detail"] += 1
            elif i < len(gt_list):
                metrics["amount"]["fn"] += 1; metrics["detail"]["fn"] += 1
            else:
                metrics["amount"]["fp"] += 1; metrics["detail"]["fp"] += 1

    # --- Calculations ---
    actual_calls = total - regex_hits
    normal_cost = (total / 1000) * price_per_1k
    opt_cost = (actual_calls / 1000) * price_per_1k
    
    p50_lat = statistics.median(latencies)
    p95_lat = sorted(latencies)[int(0.95 * total)]
    
    p_amt, r_amt, f1_amt = calculate_metrics(metrics["amount"]["tp"], metrics["amount"]["fp"], metrics["amount"]["fn"])
    p_det, r_det, f1_det = calculate_metrics(metrics["detail"]["tp"], metrics["detail"]["fp"], metrics["detail"]["fn"])

    # --- 📊 Detailed Report Output ---
    print(f"\n📊 Final Report: {model_id}")
    print(f"| Metric Group | Metric | Value |")
    print(f"| :--- | :--- | :--- |")
    print(f"| **Overall** | **Avg F1 Score** | **{(f1_amt + f1_det)/2:.4f}** |")
    print(f"| | Exact Match Rate | {(exact_matches/total)*100:.1f}% |")
    print(f"| | Count Accuracy | {(correct_counts/total)*100:.1f}% |")
    print(f"| **Field: Amount** | Precision | {p_amt:.4f} |")
    print(f"| | Recall | {r_amt:.4f} |")
    print(f"| | F1 Score | {f1_amt:.4f} |")
    print(f"| **Field: Detail** | Precision | {p_det:.4f} |")
    print(f"| | Recall | {r_det:.4f} |")
    print(f"| | F1 Score | {f1_det:.4f} |")
    print(f"| **Performance** | Avg Latency | {sum(latencies)/total:.2f}s |")
    print(f"| | **p50 Latency** | **{p50_lat:.2f}s** |")
    print(f"| | **p95 Latency** | **{p95_lat:.2f}s** |")
    print(f"| **Cost Optimization** | Normal Cost | ${normal_cost:.6f} |")
    print(f"| | Optimized Cost | ${opt_cost:.6f} |")
    print(f"| | **Total Savings** | **${normal_cost - opt_cost:.6f}** |")
    print(f"| | Regex Efficiency | {(regex_hits/total)*100:.1f}% saved |")
    
    print("\nFailure Taxonomy Summary:")
    for k, v in taxonomy.items():
        print(f"  - {k:<25}: {v}")
    print("-" * 60)

if __name__ == "__main__":
    DATASET = "data/transactions_dataset.jsonl"
    models = [
        {"id": "google/gemini-2.5-flash", "cost": 0.11},
        {"id": "openai/gpt-4o-mini", "cost": 0.15},
        {"id": "anthropic/claude-haiku-4.5", "cost": 0.003}
    ]
    for m in models:
        run_evaluation(m["id"], DATASET, price_per_1k=m["cost"])