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
    save_dir = run_dir / "save"
    export_dirs = [
        p for p in save_dir.iterdir()
        if p.is_dir() and p.name.endswith("-export")
    ]
    if not export_dirs:
        raise FileNotFoundError(f"No '*-export' dirs in {save_dir}")
    return max(export_dirs, key=lambda p: p.stat().st_mtime)


def patch_obj_mtllib(obj_path: Path, new_mtl_name: str):
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
      - Uses obj["label"] as filename base: Label.obj / Label.mtl
    - Writes meshes into temp/<run_id>/dreamfusion/
    - Updates obj["mesh"] with paths

    Returns updated pipeline_scene.
    """

    node_dir = get_node_dir(run_dir, NODE_NAME)
    df_root = config.DREAMFUSION_ROOT
    runs_root = df_root / "outputs" / "dreamfusion-if"

    for obj in pipeline_scene["objects"]:
        label: str = obj["label"]
        prompt: str = obj["prompt"]
        print(f"\n[DreamFusion] Object {obj['index']} label={label} prompt='{prompt}'")

        # 1) Choose trained run dir
        if use_existing_runs and existing_run_dirs_by_label and label in existing_run_dirs_by_label:
            # testing mode: explicit mapping
            run_folder_name = existing_run_dirs_by_label[label]
            trained_run_dir = runs_root / run_folder_name
            if not trained_run_dir.is_dir():
                raise FileNotFoundError(f"Run dir '{trained_run_dir}' not found")
            print(f"[DreamFusion] Using existing run: {trained_run_dir}")
        elif use_existing_runs:
            # fallback: pick latest run starting with label lowercased + "@"
            slug = label.lower()
            candidates = [
                p for p in runs_root.iterdir()
                if p.is_dir() and p.name.startswith(slug + "@")
            ]
            if not candidates:
                raise FileNotFoundError(f"No run dir for slug '{slug}' in {runs_root}")
            trained_run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"[DreamFusion] Using latest run for slug '{slug}': {trained_run_dir}")
        else:
            # full train+export mode (for when you have server again)
            train_cmd = [
                "python", "launch.py",
                "--config", "configs/dreamfusion-if.yaml",
                "--train",
                "--gpu", str(gpu),
                f'system.prompt_processor.prompt="{prompt}"',
                "system.background.random_aug=true",
            ]
            run_cmd(train_cmd, cwd=df_root)

            slug = label.lower()
            candidates = [
                p for p in runs_root.iterdir()
                if p.is_dir() and p.name.startswith(slug + "@")
            ]
            trained_run_dir = max(candidates, key=lambda p: p.stat().st_mtime)

        # 2) Export
        parsed_config = trained_run_dir / "configs" / "parsed.yaml"
        resume_ckpt = trained_run_dir / "ckpts" / "last.ckpt"

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
