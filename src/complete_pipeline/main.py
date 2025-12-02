# src/complete_pipeline/main.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from . import config
from .nodes.multi_agents_node import run_multi_agents_node
from .nodes.gat_node import run_gat_node
from .nodes.dreamfusion_node import run_dreamfusion_for_scene
from .nodes.compositions_node import run_composition_node


# Repo root: .../complete_pipeline/
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_FILE = ROOT_DIR / "prompt.txt"


def _make_run_dir(run_id: str) -> Path:
    """Create the temp run directory: <repo>/temp/<run_id>."""
    run_dir = config.TEMP_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_prompt_from_args(args: argparse.Namespace) -> str:
    """Resolve the user prompt from CLI args or prompt.txt."""
    # 1) Explicit prompt file
    if getattr(args, "prompt_file", None):
        path = Path(args.prompt_file)
        if not path.is_file():
            raise SystemExit(f"[PIPELINE] Prompt file not found: {path}")
        text = path.read_text(encoding="utf8").strip()
        if not text:
            raise SystemExit(f"[PIPELINE] Prompt file is empty: {path}")
        print(f"[PIPELINE] Using prompt from file: {path}")
        return text

    # 2) Inline prompt string
    if getattr(args, "prompt", None):
        print("[PIPELINE] Using prompt from --prompt argument")
        return args.prompt.strip()

    # 3) Default prompt.txt at repo root (if present)
    if DEFAULT_PROMPT_FILE.is_file():
        text = DEFAULT_PROMPT_FILE.read_text(encoding="utf8").strip()
        if text:
            print(f"[PIPELINE] Using prompt from default file: {DEFAULT_PROMPT_FILE}")
            return text

    # 4) Absolute fallback – hardcoded example
    fallback = (
        "A red sofa in a cozy living room, "
        "with a small coffee table in front of the sofa."
    )
    print("[PIPELINE] No prompt provided; using built-in fallback prompt.")
    return fallback


def run_pipeline(
    test_mode: bool = True,
    user_prompt: str | None = None,
    enrich_scene: bool = True,
) -> Dict[str, Any]:
    """
    Top-level pipeline orchestration.

    Args:
        test_mode:
            - True  -> use_existing_runs=True in DreamFusion node (expect cached runs).
            - False -> full train+export DreamFusion.
        user_prompt:
            - Scene description. If None, a default fallback is used.
        enrich_scene:
            - If True, Attribute Enrichment Agent runs.
            - If False, planner -> entity/attribute/relational/... without enrichment.
    """
    if user_prompt is None:
        user_prompt = (
            "A red sofa in a cozy living room, "
            "with a small coffee table in front of the sofa."
        )

    # ---- Create run dir ----
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _make_run_dir(run_id)

    print(f"[PIPELINE] run_id = {run_id}")
    print(f"[PIPELINE] run_dir = {run_dir}")
    print(f"[PIPELINE] test_mode = {test_mode}")
    print(f"[PIPELINE] enrich_scene = {enrich_scene}")
    print(f"[PIPELINE] prompt = {user_prompt!r}")

    # ---- 1) Multi-agent scene understanding ----
    pipeline_scene = run_multi_agents_node(
        run_dir=run_dir,
        user_prompt=user_prompt,
        enrich_scene=enrich_scene,
    )

    # ---- 2) Graph Attention Network layout prediction ----
    pipeline_scene = run_gat_node(
        run_dir=run_dir,
        pipeline_scene=pipeline_scene,
    )

    # ---- 3) DreamFusion mesh generation ----
    if test_mode:
        print("[PIPELINE] test_mode=True → using existing DreamFusion runs (no training).")
        pipeline_scene = run_dreamfusion_for_scene(
            run_dir=run_dir,
            pipeline_scene=pipeline_scene,
            gpu=0,
            use_existing_runs=True,
            existing_run_dirs_by_label=None,
        )
    else:
        print("[PIPELINE] test_mode=False → training DreamFusion for each object.")
        pipeline_scene = run_dreamfusion_for_scene(
            run_dir=run_dir,
            pipeline_scene=pipeline_scene,
            gpu=0,
            use_existing_runs=False,
            existing_run_dirs_by_label=None,
        )

    # ---- 4) Composition / assembly into final OBJ scene ----
    pipeline_scene = run_composition_node(
        run_dir=run_dir,
        pipeline_scene=pipeline_scene,
    )

    # ---- 5) Save final pipeline scene summary ----
    final_path = run_dir / "final_pipeline_scene.json"
    final_path.write_text(
        json.dumps(pipeline_scene, indent=2, ensure_ascii=False),
        encoding="utf8",
    )
    print(f"[PIPELINE] Final PipelineScene saved to: {final_path}")

    return pipeline_scene


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end 3D scene generation pipeline"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Scene description as a single string.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to a text file containing the scene description.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Enable test mode (re-use existing DreamFusion runs / "
            "skip training new ones)."
        ),
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Disable Attribute Enrichment Agent; use original prompt only.",
    )

    args = parser.parse_args()
    user_prompt = _load_prompt_from_args(args)
    enrich_scene = not args.no_enrich

    # call the orchestrator
    run_pipeline(
        test_mode=args.test,
        user_prompt=user_prompt,
        enrich_scene=enrich_scene,
    )


if __name__ == "__main__":
    main()
