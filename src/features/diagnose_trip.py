"""Why does a trip have low candidate recall?

usage: python -m src.features.diagnose_trip data/raw/train/00000033
Saves reports/figures/diag_<trip>.png and prints a short diagnosis.
"""
import pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import shapely
from src.features.loader import load_trip, to_local_xy
from src.features.features import Network, RADIUS_M


def runs(mask):
    """(start, end) of consecutive True blocks."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts, ends))


def main(trip_dir):
    tid, nodes, arcs, track, route = load_trip(trip_dir)
    nodes, track = to_local_xy(nodes, track)
    net = Network(nodes, arcs)
    x, y = track.x.to_numpy(float), track.y.to_numpy(float)
    n = len(track)

    r_idx = np.array([net.id_to_idx[a] for a in route])
    # distance of every point to its nearest ROUTE arc
    d_route = np.full(n, np.inf)
    for i in range(0, n, 2000):
        sl = slice(i, i + 2000)
        d, _, _ = net.project(x[sl, None], y[sl, None], r_idx[None, :])
        d_route[sl] = d.min(axis=1)
    # distance to the nearest arc of ANY kind
    _, d_any = net.tree.query_nearest(shapely.points(x, y), return_distance=True, all_matches=False)

    miss = d_route > RADIUS_M
    print(f"trip {tid}: {n} points, {len(route)} route arcs")
    print(f"points with no route arc within {RADIUS_M:.0f} m: {miss.sum()} ({miss.mean():.1%})")
    if "t" in track:
        dt = np.diff(track.t.to_numpy(float))
        print("time step dt quantiles (min/25/50/75/max):", np.round(np.quantile(dt, [0, .25, .5, .75, 1]), 2))
    if miss.any():
        bl = sorted(runs(miss), key=lambda b: b[0] - b[1])
        print(f"missing points form {len(bl)} blocks; longest 5 (start-end, length):")
        for s, e in bl[:5]:
            print(f"   {s}-{e}  len {e - s + 1}")
        print(f"share of missing points in first 10% / last 10% of track: "
              f"{miss[: n // 10].sum() / miss.sum():.0%} / {miss[-(n // 10):].sum() / miss.sum():.0%}")
        print(f"for missing points: median dist to nearest ANY arc = {np.median(d_any[miss]):.0f} m, "
              f"to nearest ROUTE arc = {np.median(d_route[miss]):.0f} m")
        print("  -> any-arc small, route-arc large : vehicle is on roads the route does not contain")
        print("  -> both large                      : GPS far from every road (outliers / dropout / missing map)")
    print("arc length m, route arcs median/max:", np.round([np.median(net.length[r_idx]), net.length[r_idx].max()], 1))

    # figure
    pad = 300
    x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    near = (np.maximum(net.ax, net.bx) > x0) & (np.minimum(net.ax, net.bx) < x1) & \
           (np.maximum(net.ay, net.by) > y0) & (np.minimum(net.ay, net.by) < y1)
    seg = lambda ii: np.stack([np.column_stack([net.ax[ii], net.ay[ii]]), np.column_stack([net.bx[ii], net.by[ii]])], 1)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(seg(np.flatnonzero(near)), colors="0.8", linewidths=.6))
    ax.add_collection(LineCollection(seg(r_idx), colors="tab:blue", linewidths=1.6))
    ax.scatter(x[~miss], y[~miss], s=2, c="tab:green", label="GPS (route arc within range)")
    ax.scatter(x[miss], y[miss], s=4, c="tab:red", label="GPS (no route arc within range)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title(f"trip {tid}: grey=network, blue=true route")
    out = pathlib.Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"diag_{tid}.png", dpi=130, bbox_inches="tight")
    print("saved", out / f"diag_{tid}.png")


if __name__ == "__main__":
    main(sys.argv[1])
