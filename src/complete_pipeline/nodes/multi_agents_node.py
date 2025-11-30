# src/complete_pipeline/nodes/multi_agents_node.py

from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json

from ..utils.fs import get_node_dir
from .. import config

from Multi_Agents.multi_agents import run_multi_agents_scene
from ..core.scene_pipeline import build_pipeline_scene_from_scene_graph

NODE_NAME = "multi_agents"


def run_multi_agents_node(
    run_dir: Path,
    user_prompt: str,
    enrich_scene: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline node:

    - Runs the multi-agent system with optional scene enrichment
    - Converts scene_graph into PipelineScene
    - Saves artifacts under temp/<run_id>/multi_agents/
    """
    node_dir = get_node_dir(run_dir, NODE_NAME)

    # 1) Run multi-agents workflow with chosen mode
    result = run_multi_agents_scene(user_prompt, enrich_scene=enrich_scene)
    scene_graph = result["scene_graph"]
    traces = result.get("traces", {})

    # 2) Convert to PipelineScene
    pipeline_scene = build_pipeline_scene_from_scene_graph(
        scene_graph,
        drop_environment=True,
    )

    # 3) Save debug artifacts (unchanged)
    node_dir.mkdir(parents=True, exist_ok=True)

    (node_dir / "scene_graph.json").write_text(
        json.dumps(scene_graph, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    (node_dir / "pipeline_scene.json").write_text(
        json.dumps(pipeline_scene, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    (node_dir / "traces.json").write_text(
        json.dumps(traces, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    (node_dir / "raw_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf8",
    )

    print(f"[MultiAgents] Saved scene_graph, pipeline_scene, traces to {node_dir}")

    return pipeline_scene
