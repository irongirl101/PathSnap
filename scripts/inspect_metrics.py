"""Inspect per-trip test metrics to find success/failure cases for figures."""
import json

d = json.load(open('C:/Users/Adithya Ajeeth/PathSnap/artifacts/evaluation/test_metrics.json'))
for h, c in zip(d['hmm']['per_trip'], d['classifier_only']['per_trip']):
    tid = h['trip_id']
    print(f"{tid}: HMM f1={h['edge_f1']:.3f} rmf={h['rmf']:.3f} acc={h['point_accuracy']:.3f} cov={h['coverage']:.3f} | "
          f"CLF f1={c['edge_f1']:.3f} rmf={c['rmf']:.3f} acc={c['point_accuracy']:.3f} | "
          f"HMM gaps={len(h.get('gaps', [])) if isinstance(h.get('gaps'), list) else 'n/a'} unmatched={h['unmatched_points']}")
