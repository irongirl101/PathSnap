"""Plots, geographic overlays and split-level summaries."""
from collections import defaultdict
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def map_trip(out, trip, graph, route, points):
    coords = np.array([(p.lon, p.lat) for p in points])
    road = graph.lonlat[graph.arcs]
    ref = road[route]
    stem = out / f"map_{trip.partition}_{trip.trip_id}"
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.add_collection(LineCollection(road, color="#9a9a9a", linewidth=.4, alpha=.55, label="Road network"))
    ax.add_collection(LineCollection(ref, color="#07855a", linewidth=2, label="Reference arcs"))
    ax.scatter(*coords.T, s=3, color="#1c4ba2", alpha=.6, label="GPS")
    ax.set(xlabel="Longitude (°)", ylabel="Latitude (°)", title=f"{trip.partition} · {trip.trip_id}")
    ax.set_aspect(1 / np.cos(np.radians(coords[:, 1].mean())))
    ax.autoscale_view()
    ax.legend()
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)

    try:
        import folium
    except ImportError:
        return
    m = folium.Map(location=coords.mean(axis=0)[::-1].tolist())
    for name, color, paths, width in (("Road network", "#888888", road, 1),
                                        ("Reference arcs", "#07855a", ref, 4)):
        layer = folium.FeatureGroup(name=name)
        folium.PolyLine(paths[:, :, ::-1].tolist(), color=color, weight=width).add_to(layer)
        layer.add_to(m)
    gps = folium.FeatureGroup(name="GPS observations")
    for lon, lat in coords:
        folium.CircleMarker([lat, lon], radius=2, color="#1c4ba2", fill=True).add_to(gps)
    gps.add_to(m)
    m.fit_bounds([coords.min(axis=0)[::-1].tolist(), coords.max(axis=0)[::-1].tolist()])
    folium.LayerControl().add_to(m)
    m.save(str(stem.with_suffix(".html")))


def summarize(out, summaries, evaluations):
    figures, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for partition in ("train", "student_test"):
        group = [r for r in summaries if r["partition"] == partition]
        for ax, key, title in zip(axes.flat,
                ("gps_length_m","raw_points","arcs","max_dt_s"),
                ("Recorded GPS length (m)","Raw GPS observations","Directed road arcs","Maximum GPS interval (s)")):
            ax.hist([r[key] for r in group], bins=18, alpha=.5, label=partition)
            ax.set(xlabel=title, ylabel="Trips")
            ax.legend()
    figures.savefig(out / "dataset_distributions.png", dpi=150)
    plt.close(figures)

    grouped = defaultdict(list)
    for row in evaluations:
        grouped[row["partition"], row["radius_m"], row["k"]].append(row)
    results = []
    for (partition, radius, k), rows in sorted(grouped.items()):
        def total(key):
            return sum(row[key] for row in rows)
        def ratio(a, b):
            return total(a) / total(b) if total(b) else None
        recall = ratio("accepted_hits", "accepted_points")
        results.append(dict(partition=partition, radius_m=radius, k=k,
                            points=total("points"), accepted_points=total("accepted_points"),
                            any_reference_point_rate=ratio("point_any_reference", "points"),
                            route_arc_coverage=ratio("unique_reference_hit", "unique_reference_count"),
                            provisional_recall=recall, provisional_miss_rate=None if recall is None else 1-recall,
                            empty_rate=ratio("empty", "points")))
    with (out / "candidate_metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
