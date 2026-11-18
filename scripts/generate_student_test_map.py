"""Export an actual API prediction and reference/GPS comparison for one student-test trip."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import joblib

from src.api.trips import TripStore
from src.pathsnap.api_adapter import matcher


def export_map(trip_id, data_root, model_path, output_dir):
    store = TripStore(data_root)
    trip = store.load_prediction_input(trip_id).as_dict()
    reference = store.read_reference_route(trip_id)
    model = joblib.load(model_path)
    prediction = matcher(trip, {"model": model})

    nodes = np.asarray([[node["lon"], node["lat"]] for node in trip["nodes"]], dtype=float)
    arcs = np.asarray([[arc["from_node_id"], arc["to_node_id"]] for arc in trip["arcs"]], dtype=int)
    gps = np.asarray([[point["lon"], point["lat"]] for point in trip["points"]], dtype=float)
    road_paths = nodes[arcs]
    reference_paths = road_paths[np.asarray(reference, dtype=int)]
    predicted_paths = road_paths[np.asarray(prediction["route_arc_ids"], dtype=int)]
    margin_lon = max(.005, float(np.ptp(gps[:, 0])) * .03)
    margin_lat = max(.005, float(np.ptp(gps[:, 1])) * .03)
    road_min = road_paths.min(axis=1)
    road_max = road_paths.max(axis=1)
    nearby = (
        (road_min[:, 0] <= gps[:, 0].max() + margin_lon)
        & (road_max[:, 0] >= gps[:, 0].min() - margin_lon)
        & (road_min[:, 1] <= gps[:, 1].max() + margin_lat)
        & (road_max[:, 1] >= gps[:, 1].min() - margin_lat)
    )
    context_paths = road_paths[nearby]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"map_{trip_id}"

    import folium

    center = gps.mean(axis=0)[::-1].tolist()
    map_view = folium.Map(location=center, tiles="OpenStreetMap")
    roads_layer = folium.FeatureGroup(name="Nearby road network", show=True)
    for path in context_paths:
        folium.PolyLine(path[:, ::-1].tolist(), color="#888888", weight=1, opacity=.45).add_to(roads_layer)
    roads_layer.add_to(map_view)
    layers = (
        ("Observed GPS", "#1c4ba2", gps[:, ::-1].tolist(), 3),
        ("Reference route", "#07855a",
         reference_paths[:, :, ::-1].reshape(-1, 2).tolist(), 5),
        ("Predicted route", "#d93636",
         predicted_paths[:, :, ::-1].reshape(-1, 2).tolist(), 5),
    )
    for name, color, points, width in layers:
        layer = folium.FeatureGroup(name=name, show=True)
        if name == "Observed GPS":
            folium.PolyLine(points, color=color, weight=width, opacity=.8).add_to(layer)
            for label, point in (("Start", points[0]), ("End", points[-1])):
                folium.Marker(point, tooltip=label, icon=folium.Icon(color="blue" if label == "Start" else "darkred")).add_to(layer)
        else:
            for start in range(0, len(points) - 1, 2):
                folium.PolyLine(points[start:start + 2], color=color, weight=width, opacity=.8).add_to(layer)
        layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:white;padding:8px 10px;border:1px solid #777;border-radius:4px;">'
        '<b>Map layers</b><br><span style="color:#888">■</span> Nearby roads<br>'
        '<span style="color:#1c4ba2">■</span> GPS observations<br>'
        '<span style="color:#07855a">■</span> Reference route<br>'
        '<span style="color:#d93636">■</span> Predicted route</div>'
    )
    map_view.get_root().html.add_child(folium.Element(legend))
    bounds = np.vstack([gps, reference_paths.reshape(-1, 2), predicted_paths.reshape(-1, 2)])
    map_view.fit_bounds([bounds.min(axis=0)[::-1].tolist(), bounds.max(axis=0)[::-1].tolist()])
    map_view.save(stem.with_suffix(".html"))

    fig, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    axis.add_collection(LineCollection(road_paths, color="#cccccc", linewidth=.3, alpha=.5,
                                       label="Road network"))
    axis.add_collection(LineCollection(reference_paths, color="#07855a", linewidth=1.6,
                                       label="Reference route"))
    axis.add_collection(LineCollection(predicted_paths, color="#d93636", linewidth=1.3,
                                       label="Predicted route"))
    axis.plot(gps[:, 0], gps[:, 1], color="#1c4ba2", linewidth=.7, alpha=.8, label="GPS observations")
    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"Student-test trip {trip_id}")
    axis.set_aspect(1 / np.cos(np.radians(gps[:, 1].mean())))
    axis.autoscale_view()
    axis.legend()
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"saved {stem}.html and {stem}.png; {len(context_paths)} context arcs, "
          f"predicted {len(prediction['route_arc_ids'])} arcs "
          f"for {len(gps)} GPS points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--data-root", default="data/raw/student_test")
    parser.add_argument("--model", default="artifacts/final_model.joblib")
    parser.add_argument("--output-dir", default="reports/student_test")
    args = parser.parse_args()
    export_map(args.trip_id, args.data_root, args.model, args.output_dir)
