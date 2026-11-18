"""Build data/processed/candidates.parquet from data/raw/<split>/<trip>/ folders.

usage: python -m src.features.build_dataset data/raw/train
"""
import hashlib, pathlib, sys
import numpy as np, pandas as pd
from src.features.loader import load_trip, to_local_xy
from src.features.features import Network, build_trip_table, candidate_recall, save_feature_spec


def net_key(d):
    h = hashlib.md5()
    for ext in ("nodes", "arcs"):
        h.update((d / f"{d.name}.{ext}").read_bytes())
    return h.hexdigest()


def main(root, out="data/processed/candidates.parquet"):
    root = pathlib.Path(root)
    nets, tables, summary = {}, [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        tid, nodes, arcs, track, route = load_trip(d)
        nodes_xy, track_xy = to_local_xy(nodes, track)
        key = net_key(d)                       # identical networks are built once
        if key not in nets:
            nets[key] = Network(nodes_xy, arcs)
            q = np.quantile(nets[key].length, [.5, .95, .99, 1.0])
            print(f"[network {key[:6]}] {len(nodes)} nodes, {len(arcs)} arcs; "
                  f"arc length m  median/p95/p99/max = {np.round(q, 1)}")
        net = nets[key]
        tab = build_trip_table(tid, nodes_xy, arcs, track_xy, route, net=net)
        if tab.empty:
            print(f"!! {tid}: no candidates"); continue
        tables.append(tab)
        # consistency checks
        ridx = [net.id_to_idx[a] for a in route if a in net.id_to_idx]
        connected = np.mean([net.v[a] == net.u[b] for a, b in zip(ridx[:-1], ridx[1:])]) if len(ridx) > 1 else np.nan
        step = np.hypot(np.diff(track_xy.x), np.diff(track_xy.y)) / np.diff(track_xy.t)
        pos = tab[tab.label == 1]
        summary.append(dict(trip_id=tid, n_points=len(track), n_route_arcs=len(route),
                            route_arcs_missing=len(route) - len(ridx), route_connected=connected,
                            speed_med=np.median(step), speed_p95=np.quantile(step, .95), speed_max=step.max(),
                            gps_to_route_med=pos.dist.median(), gps_to_route_p95=pos.dist.quantile(.95),
                            recall=candidate_recall(tab), n_rows=len(tab)))
    df = pd.concat(tables, ignore_index=True)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    s = pd.DataFrame(summary)
    pathlib.Path("reports").mkdir(exist_ok=True)
    s.to_csv("reports/trip_summary.csv", index=False)
    save_feature_spec()
    print(s.round(2).to_string(index=False))
    print(f"\nrows {len(df)} | positives {int(df.label.sum())} | overall candidate recall {candidate_recall(df):.3f}")
    print(f"saved {out}, reports/trip_summary.csv, artifacts/feature_columns.json")


if __name__ == "__main__":
    main(sys.argv[1])
