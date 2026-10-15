"""
AFAS - Expanded Vision Benchmark Evaluation Script
===================================================
Evaluates YOLOv8n across the combined 2,249 image dataset:
- Original clean dataset (999 images)
- Augmented environmental stress tests (800 fire images: low-light, smoke haze, motion blur/noise)
- Hard negative distractors (450 non-fire images: sunset/tungsten warmth, dark rooms, steam)
"""

import os
import glob
import time
import json
import collections
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8_fire.pt')

def run_evaluation():
    print("=" * 80)
    print("AFAS EXPANDED VISION EVALUATION BENCHMARK (N = 2,249 IMAGES)")
    print("=" * 80)

    yolo = YOLO(MODEL_PATH)

    # 1. Original Dataset (999)
    orig_fire_dir = os.path.join(BASE_DIR, 'data', 'raw', 'fire_dataset', 'fire_dataset', 'fire_images')
    orig_nf_dir = os.path.join(BASE_DIR, 'data', 'raw', 'fire_dataset', 'fire_dataset', 'non_fire_images')

    orig_fire = [(p, 1, 'Original Clean Fire') for p in glob.glob(os.path.join(orig_fire_dir, '*.*'))]
    orig_nf = [(p, 0, 'Original Clean Safe') for p in glob.glob(os.path.join(orig_nf_dir, '*.*'))]

    # 2. Expanded Environmental Dataset (1,250)
    exp_fire_dir = os.path.join(BASE_DIR, 'data', 'processed', 'expanded_test', 'fire')
    exp_nf_dir = os.path.join(BASE_DIR, 'data', 'processed', 'expanded_test', 'non_fire')

    def get_fire_cat(filename):
        if 'night' in filename: return 'Low-Light / Night Fire'
        if 'haze' in filename: return 'Dense Smoke Haze Fire'
        if 'vibe' in filename: return 'Motion Blur & Noise Fire'
        return 'Augmented Fire'

    def get_nf_cat(filename):
        if 'warm' in filename: return 'Sunset / Warm Light Distractor'
        if 'night' in filename: return 'Night Safe Room'
        if 'steam' in filename: return 'Steam / Fog Safe Room'
        return 'Augmented Safe'

    exp_fire = [(p, 1, get_fire_cat(os.path.basename(p))) for p in glob.glob(os.path.join(exp_fire_dir, '*.*'))]
    exp_nf = [(p, 0, get_nf_cat(os.path.basename(p))) for p in glob.glob(os.path.join(exp_nf_dir, '*.*'))]

    all_samples = orig_fire + orig_nf + exp_fire + exp_nf
    print(f"Total Test Samples: {len(all_samples):,}")
    print(f"  - Fire Samples    : {len(orig_fire) + len(exp_fire):,} (Original: {len(orig_fire)}, Stress: {len(exp_fire)})")
    print(f"  - Non-Fire Samples: {len(orig_nf) + len(exp_nf):,} (Original: {len(orig_nf)}, Distractors: {len(exp_nf)})")
    print("\nRunning inference...")

    results = []
    classes_detected = {'smoke': 0, 'fire': 0}

    start = time.time()
    for i, (p, label, cat) in enumerate(all_samples):
        try:
            img = Image.open(p).convert('RGB')
            res = yolo(img, verbose=False)
            max_c = 0.0
            for r in res:
                for b in r.boxes:
                    c = float(b.conf[0])
                    cls_id = int(b.cls[0])
                    cls_name = yolo.names.get(cls_id, 'fire')
                    classes_detected[cls_name] = classes_detected.get(cls_name, 0) + 1
                    if c > max_c:
                        max_c = c
            
            pred = 1 if max_c >= 0.25 else 0
            results.append({
                'path': p,
                'label': label,
                'pred': pred,
                'conf': round(max_c, 4),
                'category': cat
            })
        except Exception:
            pass

        if (i + 1) % 500 == 0 or (i + 1) == len(all_samples):
            print(f"  Processed {i+1}/{len(all_samples)} frames...")

    total_time = time.time() - start
    avg_latency = (total_time / len(results)) * 1000

    y_true = [r['label'] for r in results]
    y_pred = [r['pred'] for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Category Breakdown
    cat_stats = collections.defaultdict(lambda: {'true': [], 'pred': []})
    for r in results:
        cat_stats[r['category']]['true'].append(r['label'])
        cat_stats[r['category']]['pred'].append(r['pred'])

    print("\n" + "=" * 80)
    print(f"OVERALL PERFORMANCE METRICS (N = {len(results):,})")
    print("=" * 80)
    print(f"Total Inference Time: {total_time:.2f} s ({avg_latency:.2f} ms/frame ~ {1000/avg_latency:.1f} FPS)")
    print(f"  • Accuracy        : {acc*100:.2f}%")
    print(f"  • Precision       : {prec*100:.2f}%")
    print(f"  • Recall (Sens.)  : {rec*100:.2f}% (Catches {tp}/{len(y_true)-sum(1 for y in y_true if y==0)} fire events)")
    print(f"  • F1-Score        : {f1*100:.2f}%")
    print(f"\n[Confusion Matrix]")
    print(f"                   Actual Safe     Actual Fire")
    print(f"  Predicted Safe :     TN = {tn:<4}        FN = {fn:<4}")
    print(f"  Predicted Fire :     FP = {fp:<4}        TP = {tp:<4}")

    print("\n" + "-" * 80)
    print(f"{'Condition / Sub-Category':<32} | {'Count':<7} | {'Accuracy':<10} | {'Recall/Prec'}")
    print("-" * 80)

    category_summary = {}
    for cat, data in sorted(cat_stats.items()):
        c_acc = accuracy_score(data['true'], data['pred'])
        c_rec = recall_score(data['true'], data['pred'], zero_division=0)
        c_prec = precision_score(data['true'], data['pred'], zero_division=0)
        is_fire_cat = any(data['true'])
        key_metric = f"Recall: {c_rec*100:.1f}%" if is_fire_cat else f"Specificity: {(1 - (sum(data['pred'])/len(data['pred'])))*100:.1f}%"
        print(f"{cat:<32} | {len(data['true']):<7} | {c_acc*100:>7.2f}%   | {key_metric}")
        category_summary[cat] = {
            'count': len(data['true']),
            'accuracy': round(float(c_acc), 4),
            'recall': round(float(c_rec), 4),
            'precision': round(float(c_prec), 4)
        }

    # Save output
    output_data = {
        'total_evaluated': len(results),
        'metrics': {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'latency_ms': round(float(avg_latency), 2),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        },
        'category_breakdown': category_summary,
        'classes_detected': classes_detected
    }

    out_file = os.path.join(BASE_DIR, 'ml', 'expanded_vision_results.json')
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print("\n" + "=" * 80)
    print(f"Detailed expanded results saved to: {out_file}")
    print("=" * 80)

    return output_data

if __name__ == '__main__':
    run_evaluation()
