# src/complete_pipeline/main.py
from __future__ import annotations
from pathlib import Path

from .utils.fs import new_run_id, get_run_dir
from .nodes.dreamfusion_node import run_dreamfusion_batch
# from .nodes.gat_node import run_gat
# from .nodes.multi_agents_node import run_multi_agents
# from .nodes.compositions_node import run_compositions


def run_pipeline(test_mode: bool = True):
    run_id = new_run_id()
    run_dir = get_run_dir(run_id)
    print(f"[PIPELINE] Run id: {run_id}")
    print(f"[PIPELINE] Run dir: {run_dir}")

    # 1. DreamFusion stage
    prompts = ["a large bed", "wood chair"]

    if test_mode:
        existing_run_dirs = [
            "a_large_bed@20251122-220708",
            "a_large_bed@20251122-225933",
        ]
        df_results = run_dreamfusion_batch(
            run_dir=run_dir,
            prompts=prompts,
            gpu=0,
            use_existing_runs=True,
            existing_run_dirs=existing_run_dirs,
        )
    else:
        df_results = run_dreamfusion_batch(
            run_dir=run_dir,
            prompts=prompts,
            gpu=0,
            use_existing_runs=False,
            existing_run_dirs=None,
        )

    # Here you’d pass df_results (with OBJ/MTL paths) into the next nodes:
    # gat_results = run_gat(run_dir, df_results)
    # agents_results = run_multi_agents(run_dir, gat_results)
    # compositions_results = run_compositions(run_dir, agents_results)

    print("\n[PIPELINE] DreamFusion outputs:")
    for r in df_results:
        print(f"  - [{r['index']}] {r['prompt']}:")
        print(f"      OBJ: {r['obj_path']}")
        print(f"      MTL: {r['mtl_path']}")


if __name__ == "__main__":
    run_pipeline(test_mode=True)
