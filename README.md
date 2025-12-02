# 3D Scene Construction Pipeline

End-to-end pipeline that turns a **text scene description** into a **composed 3D scene**:

- Multi-agent LLM system → structured **scene graph**
- Graph Attention Network (GAT) → **positions & rotations**
- DreamFusion (DeepFloyd IF) → **3D meshes** per object
- Composition module → **final scene OBJ+MTL**

---

## 1. Environment & Dependencies

Tested with:

- **Python**: 3.11
- **CUDA**: 12.1
- **PyTorch**: 2.4.1 (with CUDA 12.1)

### 1.1. Create a conda env (example)

```bash
conda create -n structuralm python=3.11
conda activate structuralm
````

### 1.2. Install main requirements

From the repo root:

```bash
pip install -r requirements.txt
```

### 1.3. Install extra build-heavy dependencies

Some libs (Tiny CUDA NN, nvdiffrast, nerfacc, etc.) are tracked separately in:

* `required_git.txt`

Install them **with** `--no-build-isolation`:

```bash
pip install -r required_git.txt --no-build-isolation
```

> ⚠️ This step may take a while and requires a working CUDA toolchain.

---

## 2. Project Structure (High Level)

* `Multi_Agents/`

  * LLM-based multi-agent system (planner, entity extraction, attributes, relations, etc.)
* `Graph_Attention_Network/`

  * GNN model that predicts object transforms from the scene graph.
* `DreamFusion/`

  * DreamFusion (DeepFloyd IF) setup for text-to-3D.
  * Exports OBJ+MTL meshes per object prompt.
* `Composition/`

  * `SceneOrchestrator` that:

    * Loads OBJ/MTL assets
    * Scales/rotates/translates them from GAT output
    * Applies simple stacking / floor alignment
    * Exports a combined scene OBJ+MTL
* `src/complete_pipeline/`

  * `main.py`: pipeline entrypoint (CLI)
  * `nodes/`: glue code between components (multi-agents, GAT, DreamFusion, compositions)
  * `utils/`: helpers (filesystem paths, etc.)
* `temp/`

  * Per-run working directory: intermediate JSONs, meshes, final scene, etc. (git-ignored)

---

## 3. Configuring the Runner Script

The repo assumes you use a small wrapper script to set API keys and launch the pipeline:

* `run_template.sh` – **committed**, contains placeholders
* `run_pipeline.sh` – **git-ignored**, your local version with real keys

### 3.1. Create your local runner

From repo root:

```bash
cp run_template.sh run_pipeline.sh
chmod +x run_pipeline.sh
```

Edit `run_pipeline.sh` and fill in your keys:

```bash
export MY_LLM_API_KEY="sk-...your-key..."
export MY_LLM_BASE_URL="https://your-provider/"
export SCENE_MODEL_NAME="Llama-3.3-70B-Instruct"
export MY_HUGGINGFACE_TOKEN="hf_...your-token..."
```

At the bottom it should call:

```bash
PYTHONPATH=src:. python -m complete_pipeline.main "$@"
```

> The `"$@"` is important so CLI arguments are passed through.

---

## 4. Providing the Prompt

The pipeline reads the **scene description** from one of:

1. `--prompt "..."` (inline text)
2. `--prompt-file path/to/file.txt`
3. If neither is given, it looks for `prompt.txt` in the repo root.
4. If nothing is found, it falls back to a built-in example.

### 4.1. Easiest workflow: `prompt.txt`

Create a `prompt.txt` at the repo root:

```text
A red sofa in a cozy living room, with a small coffee table in front of the sofa.
```

Then:

```bash
./run_pipeline.sh
```

---

## 5. Running the Pipeline

### 5.1. Full mode (train DreamFusion on each object)

Requires a strong GPU with enough VRAM.

```bash
./run_pipeline.sh \
  --prompt-file prompt.txt
```

This runs:

1. **Multi-Agent System**

   * Planner → builds internal plan
   * Entity Extraction → nodes (sofa, table, etc.)
   * Attribute Extraction → color/material/style
   * Relational Inference → relations (`in_front_of`, `on_top_of`, etc.)
   * Attribute Enrichment (optional, see below)
   * Size & Scale → dimensions
   * Graph Formalization → final scene graph JSON

2. **Graph Attention Network (GAT)**

   * Predicts **position + rotation** for each object.

3. **DreamFusion**

   * For each object:

     * Uses `prompt` (e.g. `"red cozy fabric sofa"`) to train a 3D asset.
     * Saves runs under `DreamFusion/outputs/dreamfusion-if/`
     * Exports OBJ/MTL to the temp run dir:

       * `temp/<run_id>/dreamfusion/<Label>.obj`
       * `temp/<run_id>/dreamfusion/<Label>.mtl`

4. **Composition**

   * Builds a `pos.json` from the GAT output + mesh info.
   * Uses `SceneOrchestrator` to:

     * Load each DreamFusion OBJ/MTL
     * Apply scale/rotation/translation
     * Apply simple floor + stacking logic
     * Export:

       * `temp/<run_id>/compositions/scene_001.obj`
       * `temp/<run_id>/compositions/scene_001.mtl`

5. **Final summary**

   * `temp/<run_id>/final_pipeline_scene.json` contains the final structured representation plus composition metadata.

### 5.2. Test mode (re-use existing DreamFusion runs)

If you already have DreamFusion runs for your labels and just want to re-compose, you can tell the pipeline to **not train new models**, but instead re-use cached runs:

```bash
./run_pipeline.sh \
  --prompt "A red sofa in a cozy living room, with a small coffee table in front of the sofa." \
  --test
```

This sets `test_mode=True`, so `DreamFusion`:

* skips training,
* looks for matching runs in `DreamFusion/outputs/dreamfusion-if`,
* exports meshes from the latest matching run for each object label.

---

## 6. Attribute Enrichment Toggle

By default, the **Attribute Enrichment Agent** is **enabled** (it slightly elaborates the scene prompt).

To **disable** enrichment and use the original prompt as-is:

```bash
./run_pipeline.sh --prompt-file prompt.txt --no-enrich
```

Internally this calls:

```python
run_pipeline(
    test_mode=...,
    user_prompt=...,
    enrich_scene=False,
)
```

---

## 7. Outputs & Paths (Where to look)

For each run, a timestamped directory is created under:

```text
temp/<YYYY-MM-DD_HH-MM-SS>/
```

Inside:

* `multi_agents/`

  * `scene_graph.json` – final graph from LLM pipeline
  * `pipeline_scene.json` – internal PipelineScene representation
  * `raw_result.json` – full multi-agent traces and intermediate outputs
* `graph_attention_network/`

  * `scene_for_gat.json`
  * `gat_output.json`
  * `pipeline_scene_after_gat.json`
* `dreamfusion/`

  * `<Label>.obj` / `<Label>.mtl` – final per-object meshes copied from DreamFusion
* `compositions/`

  * `pos.json` – layout for the SceneOrchestrator
  * `scene_001.obj` / `scene_001.mtl` – final composed scene
  * `pipeline_scene_after_compositions.json` – pipeline scene with composition metadata
* `final_pipeline_scene.json`

  * Master summary of the entire run

---

## 8. Notes

* `run_pipeline.sh` is **not** committed; use `run_template.sh` as a safe template.
* Large outputs (e.g. DreamFusion training logs/runs, temp folders) are **git-ignored**.
* `__pycache__/` directories and other build artefacts are also ignored.

You can customize prompts, toggles, and intermediate behavior without touching the core pipeline code—just through:

* `prompt.txt` / `--prompt` / `--prompt-file`
* `--test` (DreamFusion behaviour)
* `--no-enrich` (LLM enrichment behaviour)
