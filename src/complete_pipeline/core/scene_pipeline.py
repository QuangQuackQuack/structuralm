# src/complete_pipeline/core/scene_pipeline.py

from __future__ import annotations
from typing import Dict, Any, List, Set
import copy
import re


def _build_canonical_label(raw_label: str, existing: Set[str]) -> str:
    """
    Turn arbitrary labels into a canonical, unique label suitable for
    both JSON and filenames.

    - Keep letters, digits, underscores
    - Capitalize first letter
    - Enforce uniqueness with _1, _2, ...
    """
    if not raw_label:
        raw_label = "Object"

    label = re.sub(r"[^0-9a-zA-Z_]+", "_", raw_label.strip())
    if not label:
        label = "Object"

    if label[0].isalpha():
        label = label[0].upper() + label[1:]

    base = label
    suffix = 1
    while label in existing:
        label = f"{base}_{suffix}"
        suffix += 1

    existing.add(label)
    return label


def build_pipeline_scene_from_scene_graph(
    scene_graph: Dict[str, Any],
    drop_environment: bool = True,
) -> Dict[str, Any]:
    """
    Convert your multi-agent scene_graph into a canonical PipelineScene.

    Returns:
      {
        "objects": [...],
        "relationships": [...],
        "metadata": {...},
      }
    """
    sg = copy.deepcopy(scene_graph)

    # 1) Optionally drop environment nodes
    nodes = sg.get("nodes", [])
    edges = sg.get("edges", [])

    if drop_environment:
        nodes = [n for n in nodes if n.get("category") != "environment"]
        kept_ids = {n["id"] for n in nodes}
        edges = [
            e for e in edges
            if e.get("source") in kept_ids and e.get("target") in kept_ids
        ]

    sg["nodes"] = nodes
    sg["edges"] = edges

    # 2) Build objects list
    objects: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    existing_labels: Set[str] = set()
    id_to_index: Dict[str, int] = {}

    for idx, node in enumerate(nodes):
        node_id = node.get("id")
        if not node_id:
            continue

        id_to_index[node_id] = idx

        raw_label = node.get("label") or node.get("name") or "Object"
        label = _build_canonical_label(raw_label, existing_labels)

        attrs = node.get("attributes", {}) or {}
        room_scaled = attrs.get("room_scaled_dim") or {}
        dims_m = attrs.get("dimensions_m") or {}

        def _get_dim(source: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
            val = source.get(key, fallback)
            try:
                return float(val)
            except (TypeError, ValueError):
                return fallback

        if room_scaled:
            w = _get_dim(room_scaled, "width")
            h = _get_dim(room_scaled, "height")
            d = _get_dim(room_scaled, "depth")
        else:
            w = _get_dim(dims_m, "width_m")
            h = _get_dim(dims_m, "height_m")
            d = _get_dim(dims_m, "depth_m")

        # Build DreamFusion prompt from attrs
        color = attrs.get("color")
        style = attrs.get("style")
        material = attrs.get("material")
        name = node.get("name") or raw_label

        parts = [
            p.strip()
            for p in (color, style, material)
            if isinstance(p, str) and p.strip()
        ]
        if isinstance(name, str) and name.strip():
            parts.append(name.strip())

        prompt = " ".join(parts) if parts else name

        objects.append(
            {
                "id": node_id,
                "index": idx,
                "label": label,
                "category": node.get("category"),
                "prompt": prompt,
                "bounding_box": [w, h, d],
                "center": [0.0, 0.0, 0.0],             # can be refined later
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "attributes": attrs,
                "mesh": None,
            }
        )

    # 3) Relationships
    for e in edges:
        src_id = e.get("source")
        tgt_id = e.get("target")
        rel = e.get("relation")
        if src_id not in id_to_index or tgt_id not in id_to_index or rel is None:
            continue
        relationships.append(
            {
                "source_index": id_to_index[src_id],
                "target_index": id_to_index[tgt_id],
                "source_id": src_id,
                "target_id": tgt_id,
                "relation": rel,
            }
        )

    return {
        "objects": objects,
        "relationships": relationships,
        "metadata": sg.get("metadata", {}),
    }
