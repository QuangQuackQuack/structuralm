# src/complete_pipeline/nodes/compositions_node.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import sys

from ..utils.fs import get_node_dir
from .. import config

NODE_NAME = "compositions"


def build_pos_json_from_pipeline_scene(
    pipeline_scene: Dict[str, Any],
    scene_id: str = "scene_001",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert PipelineScene into the JSON format expected by SceneOrchestrator:

    {
      "scene_001": [
        {
          "label": "Chair",
          "bounding_box": [...],
          "rotation": [...],
          "center": [...]
        },
        ...
      ]
    }
    """
    objects_list: List[Dict[str, Any]] = []

    for obj in pipeline_scene["objects"]:
        # We assume GAT has already filled center & rotation.
        objects_list.append(
            {
                "label": obj["label"],
                "bounding_box": obj["bounding_box"],
                "rotation": obj.get("rotation", [1.0, 0.0, 0.0, 0.0]),
                "center": obj.get("center", [0.0, 0.0, 0.0]),
            }
        )

    return {scene_id: objects_list}


def run_composition_node(
    run_dir: Path,
    pipeline_scene: Dict[str, Any],
    scene_id: str = "scene_001",
    include_floor: bool = True,
    objects_folder: Path | None = None,
) -> Dict[str, Any]:
    """
    Final pipeline node: takes GAT+DreamFusion outputs and produces a
    single composed scene OBJ+MTL using SceneOrchestrator.

    Steps:
      1. Build pos.json from PipelineScene
      2. Point to folder containing object OBJs/MTLs (from DreamFusion node)
      3. Load via SceneOrchestrator
      4. Export scene_001.obj / .mtl into temp/<run_id>/compositions
    """
    node_dir = get_node_dir(run_dir, NODE_NAME)
    node_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build pos.json structure from PipelineScene
    pos_data = build_pos_json_from_pipeline_scene(pipeline_scene, scene_id=scene_id)
    pos_json_path = node_dir / "pos.json"
    pos_json_path.write_text(
        json.dumps(pos_data, indent=2, ensure_ascii=False),
        encoding="utf8",
    )
    print(f"[Compositions] Wrote pos.json -> {pos_json_path}")

    # 2) Decide where object .obj/.mtl live
    # Default: DreamFusion node output dir for this run
    if objects_folder is None:
        objects_folder = get_node_dir(run_dir, "dreamfusion")
    objects_folder = Path(objects_folder)
    print(f"[Compositions] Using objects folder: {objects_folder}")

    # 3) Import SceneOrchestrator from your Compositions/composition.py
    sys.path.append(str(config.COMPOSITIONS_ROOT))
    from Composition.composition import SceneOrchestrator  # type: ignore

    orchestrator = SceneOrchestrator()

    # 4) Load scene metadata + per-object meshes
    orchestrator.load_scene_from_json(
        json_path=str(pos_json_path),
        obj_folder=str(objects_folder),
    )

    # 5) Export single scene OBJ+MTL into this node's dir
    output_obj_path = node_dir / f"{scene_id}.obj"
    orchestrator.export_scene(
        scene_id=scene_id,
        output_path=str(output_obj_path),
        include_floor=include_floor,
    )

    print(f"[Compositions] Exported composed scene -> {output_obj_path}")

    # Optionally: store path back into pipeline_scene metadata
    pipeline_scene.setdefault("composition", {})
    pipeline_scene["composition"]["scene_id"] = scene_id
    pipeline_scene["composition"]["output_obj"] = str(output_obj_path)
    pipeline_scene["composition"]["output_mtl"] = str(output_obj_path.with_suffix(".mtl"))
    pipeline_scene["composition"]["pos_json"] = str(pos_json_path)
    pipeline_scene["composition"]["objects_folder"] = str(objects_folder)

    # Save updated pipeline_scene snapshot for rewind/debug
    updated_scene_path = node_dir / "pipeline_scene_after_compositions.json"
    updated_scene_path.write_text(
        json.dumps(pipeline_scene, indent=2, ensure_ascii=False),
        encoding="utf8",
    )
    print(f"[Compositions] Saved updated PipelineScene -> {updated_scene_path}")

    return pipeline_scene
