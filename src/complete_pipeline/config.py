# src/complete_pipeline/config.py
from pathlib import Path

# Repository root is one level up from src/
REPO_ROOT = Path(__file__).resolve().parents[2]

CACHE_ROOT = REPO_ROOT / "temp"

# Where external components live (your existing repos)
DREAMFUSION_ROOT = REPO_ROOT / "DreamFusion"
GAT_ROOT = REPO_ROOT / "Graph_Attention_Network"
MULTI_AGENTS_ROOT = REPO_ROOT / "Multi_Agents"
COMPOSITIONS_ROOT = REPO_ROOT / "Compositions"