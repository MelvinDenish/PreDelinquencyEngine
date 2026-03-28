import json

d = json.load(open("test_results/scoring_segment_analysis.json"))

print("=== OVERALL METRICS ===")
for m, v in d["overall"].items():
    print(f"  {m}: AUC={v['auc_roc']}, Gini={v['gini']}, KS={v['ks_stat']}, F1={v['f1']}, Acc={v['accuracy']}, Brier={v['brier']}")

print(f"\nOptimal Threshold: {d['optimal_threshold']}")
print(f"Dataset: {d['dataset_size']} | Test: {d['test_size']}")

print("\n=== SEGMENT ANALYSIS ===")
for seg, v in d["segment_analysis"]["segment"].items():
    print(f"  {seg:20s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')} F1={v.get('f1','N/A')} Prec={v.get('precision','N/A')} Rec={v.get('recall','N/A')}")

print("\n=== AGE BANDS ===")
for ab, v in d["segment_analysis"]["age_band"].items():
    print(f"  {ab:12s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")

print("\n=== GENDER ===")
for g, v in d["segment_analysis"]["gender"].items():
    print(f"  {g:12s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")

print("\n=== REGION ===")
for r, v in d["segment_analysis"]["region"].items():
    print(f"  {r:12s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")

print("\n=== INCOME ===")
for inc, v in d["segment_analysis"]["income"].items():
    print(f"  {inc:15s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")

print("\n=== FAIRNESS ===")
print(f"  Gender DI: {d['fairness']['gender_disparate_impact']}")
print(f"  Gender Fair: {d['fairness']['gender_fair']}")
print(f"  Gender rates: {d['fairness']['gender_positive_rates']}")

if "farmer_seasonal" in d["segment_analysis"]:
    print("\n=== FARMER SEASONAL ===")
    for s, v in d["segment_analysis"]["farmer_seasonal"].items():
        print(f"  {s:15s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")

if "cross_segment_region" in d["segment_analysis"]:
    print("\n=== CROSS SEGMENT x REGION ===")
    for k, v in d["segment_analysis"]["cross_segment_region"].items():
        print(f"  {k:30s} n={v['n']:5d} AUC={v.get('auc_roc','N/A')}")
