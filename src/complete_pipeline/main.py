from __future__ import annotations
from pathlib import Path
from pprint import pprint

from .utils.fs import new_run_id, get_run_dir
from .nodes.multi_agents_node import run_multi_agents_node
from .nodes.gat_node import run_gat_node
from .nodes.compositions_node import run_composition_node
from .nodes.dreamfusion_node import run_dreamfusion_for_scene  # when server ready


def run_pipeline(test_mode: bool = True):
    run_id = new_run_id()
    run_dir = get_run_dir(run_id)
    print(f"[PIPELINE] run_id = {run_id}")
    print(f"[PIPELINE] run_dir = {run_dir}")

    user_prompt = (
        "A red sofa in a cozy living room,"
        "with a small coffee table in front of the sofa."
    )

    # 1) Multi-agents → scene_graph → PipelineScene
    pipeline_scene = run_multi_agents_node(run_dir, user_prompt, enrich_scene=False)

    # 2) GAT layout → centers & rotations
    pipeline_scene = run_gat_node(run_dir, pipeline_scene)

    # 3) DreamFusion (later, when server is available)
    if not test_mode:
        pipeline_scene = run_dreamfusion_for_scene(
            run_dir=run_dir,
            pipeline_scene=pipeline_scene,
            gpu=0,
            use_existing_runs=False,
        )

    # 4) Compositions → final combined scene OBJ+MTL
    pipeline_scene = run_composition_node(
        run_dir=run_dir,
        pipeline_scene=pipeline_scene,
        scene_id="scene_001",
        include_floor=True,
    )

    # Small human-readable summary of composition input
    composition_input = {
        "scene": [
            {
                "label": obj["label"],
                "bounding_box": obj["bounding_box"],
                "rotation": obj["rotation"],
                "center": obj["center"],
            }
            for obj in pipeline_scene["objects"]
        ]
    }

    print("\n[PIPELINE] Composition input JSON-style (for debugging):")
    pprint(composition_input)

    print("\n[PIPELINE] Final scene OBJ:", pipeline_scene["composition"]["output_obj"])
    print("[PIPELINE] Final scene MTL:", pipeline_scene["composition"]["output_mtl"])

    return {
        "pipeline_scene": pipeline_scene,
        "composition_input": composition_input,
    }


if __name__ == "__main__":
    run_pipeline(test_mode=False)
