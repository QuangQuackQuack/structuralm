# src/complete_pipeline/nodes/gat_node.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import json
import subprocess

from ..utils.fs import get_node_dir
from .. import config

NODE_NAME = "graph_attention_network"


def run_cmd(cmd, cwd: Optional[Path] = None):
    print("\n[GAT] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def build_gat_scene_from_pipeline_scene(pipeline_scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PipelineScene into the scene.json format that inference.py expects.
    """
    objects = []
    for obj in pipeline_scene["objects"]:
        idx = int(obj["index"])
        label = obj["label"]

        objects.append(
            {
                "id": idx,
                "name": f"{label}_{idx}",          # e.g. "Chair_1"
                "label": label,
                "normalized_bounding_box": obj["bounding_box"],
                "normalized_relative_center": obj.get("center", [0.0, 0.0, 0.0]),
                "rot": obj.get("rotation", [1.0, 0.0, 0.0, 0.0]),
            }
        )

    relationships = []
    for rel in pipeline_scene["relationships"]:
        relationships.append(
            {
                "obj_id1": int(rel["source_index"]),
                "obj_id2": int(rel["target_index"]),
                "relation": rel["relation"],
            }
        )

    return {
        "objects": objects,
        "relationships": relationships,
    }


def apply_gat_output_to_pipeline_scene(
    pipeline_scene: Dict[str, Any],
    gat_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Take GAT's output JSON (predicted positions & rotations) and
    write them back into pipeline_scene["objects"][i]["center"/"rotation"].
    """
    predictions = gat_output.get("predictions", [])

    # Map id -> prediction
    preds_by_id = {p["id"]: p for p in predictions}

    for obj in pipeline_scene["objects"]:
        idx = int(obj["index"])
        pred = preds_by_id.get(idx)
        if not pred:
            continue

        pred_pos = pred["prediction"]["position"]
        pred_rot = pred["prediction"]["rotation_quaternion"]

        obj["center"] = pred_pos
        obj["rotation"] = pred_rot

    return pipeline_scene


def run_gat_node(
    run_dir: Path,
    pipeline_scene: Dict[str, Any],
    checkpoint_rel: str = "checkpoints/model_phase1234.pt",
    config_rel: str = "configs/configs_syn.yaml",
) -> Dict[str, Any]:
    """
    Pipeline node for Graph Attention Network layout inference.

    Steps:
      1. Build scene_for_gat.json from PipelineScene
      2. Run inference.py (as a sub-process)
      3. Load gat_output.json
      4. Update PipelineScene with predicted positions/rotations
      5. Save all artifacts under temp/<run_id>/graph_attention_network/

    Returns updated PipelineScene.
    """
    node_dir = get_node_dir(run_dir, NODE_NAME)
    node_dir.mkdir(parents=True, exist_ok=True)

    gat_root = config.GAT_ROOT

    # 1) Build scene JSON for GAT
    scene_for_gat = build_gat_scene_from_pipeline_scene(pipeline_scene)
    scene_json_path = node_dir / "scene_for_gat.json"
    scene_json_path.write_text(
        json.dumps(scene_for_gat, indent=2, ensure_ascii=False),
        encoding="utf8",
    )
    print(f"[GAT] Wrote scene_for_gat.json -> {scene_json_path}")

    # 2) Run inference.py as in your z_run_inference.py
    checkpoint_path = gat_root / checkpoint_rel
    config_path = gat_root / config_rel
    output_json_path = node_dir / "gat_output.json"

    cmd = [
        "python",
        "inference.py",
        "--scene", str(scene_json_path),
        "--checkpoint", str(checkpoint_path),
        "--config", str(config_path),
        "--output", str(output_json_path),
    ]
    run_cmd(cmd, cwd=gat_root)

    # 3) Load GAT output JSON
    gat_output = json.loads(output_json_path.read_text(encoding="utf8"))
    print(f"[GAT] Loaded predictions from {output_json_path}")

    # 4) Apply back into pipeline_scene
    pipeline_scene = apply_gat_output_to_pipeline_scene(pipeline_scene, gat_output)

    # 5) Save updated PipelineScene for debugging / rewind
    updated_scene_path = node_dir / "pipeline_scene_after_gat.json"
    updated_scene_path.write_text(
        json.dumps(pipeline_scene, indent=2, ensure_ascii=False),
        encoding="utf8",
    )
    print(f"[GAT] Saved updated PipelineScene -> {updated_scene_path}")

    return pipeline_scene
