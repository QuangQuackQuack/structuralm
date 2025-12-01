# src/complete_pipeline/nodes/dreamfusion_node.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

from ..utils.fs import get_node_dir
from .. import config

NODE_NAME = "dreamfusion"


def run_cmd(cmd, cwd: Optional[Path] = None):
    print("\n[DreamFusion] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def find_latest_export_dir(run_dir: Path) -> Path:
    """
    Inside a DreamFusion run directory, look under 'save/' for the latest
    '*-export' folder, e.g. 'it2000-export'.
    """
    save_dir = run_dir / "save"
    if not save_dir.is_dir():
        raise FileNotFoundError(f"[DreamFusion] No 'save' dir in run: {run_dir}")

    export_dirs = [
        p for p in save_dir.iterdir()
        if p.is_dir() and p.name.endswith("-export")
    ]
    if not export_dirs:
        raise FileNotFoundError(
            f"[DreamFusion] No '*-export' dirs in {save_dir}. "
            "Make sure mesh export is enabled."
        )
    latest = max(export_dirs, key=lambda p: p.stat().st_mtime)
    print(f"[DreamFusion] Latest export dir: {latest}")
    return latest


def patch_obj_mtllib(obj_path: Path, new_mtl_name: str):
    """
    Ensure the OBJ points to the local MTL filename we copy next to it.
    """
    text = obj_path.read_text()
    lines = text.splitlines()
    new_lines = []
    replaced = False
    for line in lines:
        if line.startswith("mtllib ") and not replaced:
            new_lines.append(f"mtllib {new_mtl_name}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.insert(0, f"mtllib {new_mtl_name}")
    obj_path.write_text("\n".join(new_lines) + "\n")


def slugify_prompt(prompt: str) -> str:
    """
    Approximate the same slug DreamFusion uses for the run tag:
    - strip
    - lowercase
    - spaces -> '_'
    - replace path-unsafe chars with '_'
    """
    s = prompt.strip().lower()
    for ch in [" ", "/", "\\", ":", "?", "#", "@"]:
        s = s.replace(ch, "_")
    # collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s


def pick_latest_run(runs_root: Path) -> Path:
    """
    Pick the newest DreamFusion run directory, e.g. 'red_cozy_sofa@20251201-151545'.
    We assume no concurrent DreamFusion jobs; the latest is the one we just trained.
    """
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"[DreamFusion] Runs root does not exist: {runs_root}. "
            "Check DREAMFUSION_ROOT and exp_root_dir in configs/dreamfusion-if.yaml."
        )

    candidates = [p for p in runs_root.glob("*@*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"[DreamFusion] No runs found under {runs_root}. "
            "Did DreamFusion training finish correctly?"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[DreamFusion] Using latest run dir: {latest}")
    return latest


def pick_run_by_prompt(runs_root: Path, prompt: str) -> Path:
    """
    For reuse mode: find the most recent run whose name starts with the slugified prompt.
    """
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"[DreamFusion] Runs root does not exist: {runs_root} "
            "(check DREAMFUSION_ROOT / outputs / dreamfusion-if)."
        )

    slug = slugify_prompt(prompt)
    candidates = [
        p for p in runs_root.iterdir()
        if p.is_dir() and p.name.startswith(slug + "@")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"[DreamFusion] No runs found for prompt slug '{slug}' in {runs_root}.\n"
            f"  Prompt was: {prompt!r}"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[DreamFusion] Using run for prompt slug '{slug}': {latest}")
    return latest


def run_dreamfusion_for_scene(
    run_dir: Path,
    pipeline_scene: Dict[str, Any],
    gpu: int = 0,
    use_existing_runs: bool = False,
    existing_run_dirs_by_label: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Node:

    - Iterates over pipeline_scene["objects"]
    - For each object:
      - Uses obj["prompt"] as DreamFusion prompt
      - Writes meshes into temp/<run_id>/dreamfusion/
      - Updates obj["mesh"] with OBJ/MTL paths

    Returns updated pipeline_scene.
    """

    node_dir = get_node_dir(run_dir, NODE_NAME)
    df_root = config.DREAMFUSION_ROOT          # e.g. /.../structuralm/DreamFusion
    runs_root = df_root / "outputs" / "dreamfusion-if"

    print(f"[DreamFusion] Runs root: {runs_root}")

    for obj in pipeline_scene["objects"]:
        label: str = obj["label"]
        prompt: str = obj["prompt"]
        print(f"\n[DreamFusion] Object {obj['index']} label={label} prompt='{prompt}'")

        # 1) Choose trained run dir
        if use_existing_runs and existing_run_dirs_by_label and label in existing_run_dirs_by_label:
            # explicit mapping, e.g. {"Bed": "a_large_bed@2025..."}
            run_folder_name = existing_run_dirs_by_label[label]
            trained_run_dir = runs_root / run_folder_name
            if not trained_run_dir.is_dir():
                raise FileNotFoundError(f"[DreamFusion] Run dir '{trained_run_dir}' not found")
            print(f"[DreamFusion] Using existing run (mapping): {trained_run_dir}")

        elif use_existing_runs:
            # reuse mode without explicit mapping -> match by prompt slug
            trained_run_dir = pick_run_by_prompt(runs_root, prompt)

        else:
            # full train+export mode
            train_cmd = [
                "python", "launch.py",
                "--config", "configs/dreamfusion-if.yaml",
                "--train",
                "--gpu", str(gpu),
                f'system.prompt_processor.prompt="{prompt}"',
                "system.background.random_aug=true",
            ]
            run_cmd(train_cmd, cwd=df_root)

            # After training finishes, pick the most recent run regardless of name
            trained_run_dir = pick_latest_run(runs_root)

        # 2) Export mesh from this run
        parsed_config = trained_run_dir / "configs" / "parsed.yaml"
        resume_ckpt = trained_run_dir / "ckpts" / "last.ckpt"

        if not parsed_config.is_file():
            raise FileNotFoundError(f"[DreamFusion] parsed.yaml not found: {parsed_config}")
        if not resume_ckpt.is_file():
            raise FileNotFoundError(f"[DreamFusion] last.ckpt not found: {resume_ckpt}")

        export_cmd = [
            "python", "launch.py",
            "--config", str(parsed_config),
            "--export",
            "--gpu", str(gpu),
            f"resume={resume_ckpt}",
            "system.exporter_type=mesh-exporter",
            "system.exporter.context_type=cuda",
            "system.exporter.fmt=obj-mtl",
        ]
        run_cmd(export_cmd, cwd=df_root)

        export_dir = find_latest_export_dir(trained_run_dir)
        src_obj = export_dir / "model.obj"
        src_mtl = export_dir / "model.mtl"

        if not src_obj.is_file():
            raise FileNotFoundError(f"[DreamFusion] model.obj not found in {export_dir}")
        if not src_mtl.is_file():
            raise FileNotFoundError(f"[DreamFusion] model.mtl not found in {export_dir}")

        dst_obj = node_dir / f"{label}.obj"
        dst_mtl = node_dir / f"{label}.mtl"

        dst_obj.write_bytes(src_obj.read_bytes())
        dst_mtl.write_bytes(src_mtl.read_bytes())
        patch_obj_mtllib(dst_obj, dst_mtl.name)

        obj["mesh"] = {
            "obj": str(dst_obj),
            "mtl": str(dst_mtl),
        }

        print(f"[DreamFusion] Cached mesh for {label}:")
        print(f"  OBJ: {dst_obj}")
        print(f"  MTL: {dst_mtl}")

    return pipeline_scene
