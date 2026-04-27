import json
import time
import os
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

def run_evaluation(model_name, dataset_path, price_per_1k=0.11):
    try:
        ner = NERSystem(model_name=model_name)
    except ValueError as e:
        print(e)
        return

    # ตัวนับสำหรับวัดผลในระดับฟิลด์
    metrics = {
        "amount": {"tp": 0, "fp": 0, "fn": 0},
        "detail": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    # การจำแนกประเภทความผิดพลาด (Failure Taxonomy)
    failure_taxonomy = {
        "missed_transaction": 0,
        "hallucinated_transaction": 0,
        "wrong_amount": 0,
        "wrong_detail": 0
    }

    latencies = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ ไม่พบไฟล์ Dataset: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    print(f"\n🚀 Evaluating Model: {model_name}")
    for sample in tqdm(samples):
        text = sample['text']
        gt_list = sample['transactions']
        
        start_time = time.time()
        response = ner.parse(text)
        latencies.append(time.time() - start_time)
        
        pred_list = [t.model_dump() for t in response.transactions]
        
        # --- วิเคราะห์จำนวน Transaction ---
        if len(pred_list) < len(gt_list):
            failure_taxonomy["missed_transaction"] += (len(gt_list) - len(pred_list))
        elif len(pred_list) > len(gt_list):
            failure_taxonomy["hallucinated_transaction"] += (len(pred_list) - len(gt_list))

        # --- วิเคราะห์ความถูกต้องรายฟิลด์ ---
        for i in range(max(len(gt_list), len(pred_list))):
            if i < len(gt_list) and i < len(pred_list):
                # ตรวจสอบยอดเงิน (Amount)
                if pred_list[i]['amount'] == gt_list[i]['amount']:
                    metrics["amount"]["tp"] += 1
                else:
                    metrics["amount"]["fp"] += 1
                    failure_taxonomy["wrong_amount"] += 1
                
                # ตรวจสอบรายละเอียด (Detail)
                if str(pred_list[i]['detail']).strip() == str(gt_list[i]['detail']).strip():
                    metrics["detail"]["tp"] += 1
                else:
                    metrics["detail"]["fp"] += 1
                    failure_taxonomy["wrong_detail"] += 1
            elif i < len(gt_list):
                metrics["amount"]["fn"] += 1
                metrics["detail"]["fn"] += 1
            else:
                metrics["amount"]["fp"] += 1
                metrics["detail"]["fp"] += 1

    # --- สรุปผลลัพธ์ ---
    print("\n" + "="*40)
    print(f"  REPORT: {model_name.upper()}")
    print("="*40)
    for field in ["amount", "detail"]:
        p, r, f1 = calculate_metrics(metrics[field]["tp"], metrics[field]["fp"], metrics[field]["fn"])
        print(f"Field [{field:6}]: F1={f1:.2f} | Precision={p:.2f} | Recall={r:.2f}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
    
    print(f"\n⏱ Latency: Avg={avg_latency:.2f}s | p95={p95_latency:.2f}s")
    print(f"💰 Cost / 1k msgs: ${price_per_1k:.4f}")
    print("\n❌ Failure Taxonomy Summary:")
    for error, count in failure_taxonomy.items():
        print(f"  - {error:25}: {count}")

if __name__ == "__main__":
    DATASET_PATH = "data/transactions_dataset.jsonl"
    
    # รันการประเมินเพื่อเปรียบเทียบ 2 โมเดลตามโจทย์กำหนด
    models = [
        {"id": "google/gemini-2.5-flash", "cost": 0.11},
        {"id": "openai/gpt-4o-mini", "cost": 0.15}
    ]
    
    for m in models:
        run_evaluation(m["id"], DATASET_PATH, price_per_1k=m["cost"])
        print("\n" + "-"*40)