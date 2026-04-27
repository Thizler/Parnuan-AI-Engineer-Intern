import json
import time
import os
import pandas as pd
from tqdm import tqdm
from ner import NERSystem
from dotenv import load_dotenv

load_dotenv()

def calculate_metrics(tp, fp, fn):
    """คำนวณค่า Precision, Recall และ F1 Score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def run_evaluation(model_name, dataset_path, price_per_1k=0.11):
    try:
        ner = NERSystem(model_name=model_name)
    except ValueError as e:
        print(e)
        return

    metrics = {"amount": {"tp": 0, "fp": 0, "fn": 0}, "detail": {"tp": 0, "fp": 0, "fn": 0}}
    failure_taxonomy = {"missed": 0, "hallucinated": 0, "wrong_amount": 0, "wrong_detail": 0}
    
    error_logs = [] 
    latencies = []
    regex_hits = 0 

    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    total_samples = len(samples)
    print(f"\n🔍 Evaluating: {model_name}")
    
    for sample in tqdm(samples):
        text = sample['text']
        gt_list = sample['transactions']
        
        # ตรวจสอบประสิทธิภาพ Regex (Cost Optimization)
        if ner.parse_with_regex(text) is not None:
            regex_hits += 1

        start_time = time.time()
        response = ner.parse(text)
        latencies.append(time.time() - start_time)
        
        pred_list = [t.model_dump() for t in response.transactions]
        
        # --- Logic การเปรียบเทียบผลลัพธ์ ---
        has_error = False
        if len(pred_list) != len(gt_list):
            has_error = True
            if len(pred_list) < len(gt_list): failure_taxonomy["missed"] += 1
            else: failure_taxonomy["hallucinated"] += 1

        for i in range(max(len(gt_list), len(pred_list))):
            if i < len(gt_list) and i < len(pred_list):
                if pred_list[i]['amount'] == gt_list[i]['amount']: 
                    metrics["amount"]["tp"] += 1
                else: 
                    metrics["amount"]["fp"] += 1
                    failure_taxonomy["wrong_amount"] += 1
                    has_error = True
                
                if str(pred_list[i]['detail']).strip() == str(gt_list[i]['detail']).strip(): 
                    metrics["detail"]["tp"] += 1
                else: 
                    metrics["detail"]["fp"] += 1
                    failure_taxonomy["wrong_detail"] += 1
                    has_error = True
            elif i < len(gt_list):
                metrics["amount"]["fn"] += 1
                metrics["detail"]["fn"] += 1
                has_error = True
            else:
                metrics["amount"]["fp"] += 1
                metrics["detail"]["fp"] += 1
                has_error = True
        
        if has_error:
            error_logs.append({"text": text, "expected": gt_list, "predicted": pred_list})

    # --- 💰 คำนวณต้นทุน (Financial Analytics) ---
    actual_llm_calls = total_samples - regex_hits
    normal_cost = (total_samples / 1000) * price_per_1k
    optimized_cost = (actual_llm_calls / 1000) * price_per_1k
    savings_usd = normal_cost - optimized_cost
    savings_pct = (regex_hits / total_samples) * 100

    # --- คำนวณ Latency Metrics ---
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    # คำนวณ p95: เรียงลำดับเวลาแล้วหยิบค่าที่ตำแหน่ง 95%
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0

    # --- สรุปผล Metrics ---
    p_amt, r_amt, f1_amt = calculate_metrics(metrics["amount"]["tp"], metrics["amount"]["fp"], metrics["amount"]["fn"])
    p_det, r_det, f1_det = calculate_metrics(metrics["detail"]["tp"], metrics["detail"]["fp"], metrics["detail"]["fn"])
    avg_f1 = (f1_amt + f1_det) / 2

    # --- แสดงผลลัพธ์ในรูปแบบ Markdown Table ---
    print(f"\n📊 Result for {model_name}:")
    print(f"| Metric | Value |")
    print(f"| :--- | :--- |")
    print(f"| **Avg F1 Score** | **{avg_f1:.4f}** |")
    print(f"| Avg Latency | {avg_lat:.2f}s |")
    print(f"| **p95 Latency** | **{p95_lat:.2f}s** |") # เพิ่มส่วน p95 เข้าไป
    print(f"| Normal Cost (No Regex) | ${normal_cost:.6f} |")
    print(f"| Optimized Cost | ${optimized_cost:.6f} |")
    print(f"| **Total Savings ($)** | **${savings_usd:.6f}** |")
    print(f"| Regex Efficiency | {savings_pct:.1f}% saved |")
    
    if error_logs:
        print("\n❌ Samples of Failures (for your README):")
        for err in error_logs[:2]: 
            print(f"- Text: '{err['text']}'")
            print(f"  Exp: {err['expected']} | Pred: {err['predicted']}")

    return {
        "model": model_name, 
        "f1": avg_f1, 
        "latency": avg_lat,
        "p95_latency": p95_lat,
        "savings_usd": savings_usd,
        "optimized_cost": optimized_cost
    }

if __name__ == "__main__":
    DATASET = "data/transactions_dataset.jsonl"
    results = []
    
    models = [
        {"id": "google/gemini-2.5-flash", "cost": 0.11}, # ราคาต่อ 1k messages
        {"id": "openai/gpt-4o-mini", "cost": 0.15}
    ]
    
    for m in models:
        res = run_evaluation(m["id"], DATASET, price_per_1k=m["cost"])
        results.append(res)