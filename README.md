> [!CAUTION]
> This is the active **development branch** (`develop`) of AutoRecLab v2.
>
> - Features can change at any time.
> - Interfaces and prompts may be unstable.
> - Experimental behavior is expected.
>
> For the latest stable release (v1), use the [`main`](../../tree/main) branch.

### Prerequisites: Embeddings

This version of AutoRecLab requires pre-generated embeddings for LensKit, RecBole, and OmniRec libraries.
For the Docker container to function correctly, these files must be placed in the project's working directory.

**Setup Instructions:**

1. **Download the embeddings:** Get the files from [this link](https://1drv.ms/f/c/04375478f480c0d7/IgCMRP8a-P6yTKIUflpDpq0zAQGu9UPOn0rt_0VydNMC0iY?e=MyC9Yz). You can either download them as a ZIP archive and extract them, or download the folders directly.
2. **Create the directory:** Inside your AutoRecLab working directory, create a new folder named `ragEmbeddings`.
3. **Move the files:** Place the `lenskit`, `recbole`, and `omnirec` folders (which contain the `.pkl` and `.faiss` files) into the `ragEmbeddings` directory.

```text
AutoRecLab/
├── ragEmbeddings/
│   ├── lenskit/
│   │   ├── ... (.pkl and .faiss files)
│   ├── omnirec/
│   │   ├── ... (.pkl and .faiss files)
│   └── recbole/
│       └── ... (.pkl and .faiss files)
├── Dockerfile
├── docker-compose.yml
└── ... (other AutoRecLab files)
```
> [!CAUTION]
If errors occur while building the docker container it is most likely an issue regarding the docker-entrypoint.sh, which generates the ragEmbeddings automatically while building the container if they are not contained in your Working Directory.
In that case, delete the line "ENTRYPOINT ["/app/docker-entrypoint.sh"]" from the Dockerfile.


# AutoRecLab v2 (Develop): Towards an Autonomous Recommender-Systems Researcher

AutoRecLab is an autonomous research agent for recommender-systems experimentation.
It turns a natural-language research task into executable code, evaluates intermediate results, and improves solutions iteratively via tree search.

This `develop` branch is where new features are integrated continuously between paper releases.

## Why this branch exists

Your project follows a publication-driven release process:

- `develop`: active development, frequent changes, newest features
- `main`: stable snapshot that is updated when a new releases are considered

If you need reproducible, publication-stable behavior, use `main`.
If you want the newest capabilities, use `develop`.

## Current v2 focus (develop)

- Autonomous iterative code improvement with tree search
- Requirement engineering from free-form research prompts
- Built-in execution, scoring, and debugging loops
- RAG-assisted documentation lookup (OmniRec, LensKit, RecBole) via FAISS indices
- Configurable model/runtime behavior via `config.toml` and environment variables
- Python type checking before code execution to improve a reliable execution of the generated code

## Requirements

- Python >= 3.12
- One of:
  - uv (https://docs.astral.sh/uv/) (recommended)
  - Docker + Docker Compose
  - pip (works, but uv is preferred in this project)
- Graphviz (`dot`) available on `PATH` (required by runtime checks)
- OpenAI API key (needed for LLM calls and embedding generation)

## Quick start

### Option A: Docker (recommended for isolated runs)

1. Create `.env` in the repository root:

	```env
	OPENAI_API_KEY=your-key-here
	```

2. Run the sandbox container:

	```bash
	docker compose run --build sandbox
	```

Notes:
- The container entrypoint generates/updates documentation embeddings on startup.
- Outputs are written to `./sandbox` on the host (mounted to `/app/out` in the container).

### Option B: Local with uv

1. Install dependencies:

	```bash
	uv sync
	```

2. Create `.env` (same as above) or export your API key in the shell.

3. (Recommended once) Generate documentation embeddings locally:

	```bash
	uv run python -m cli.embeddings.main generate --all
	```

4. Start AutoRecLab:

	```bash
	uv run main.py
	```

### Option C: Local with pip

```bash
pip install -e .
python main.py
```

## Running the agent

After start, enter a multi-line research task and finish with `!start`:

```text
Enter you request, write "!start" to start:
> Build a reproducible top-N recommendation experiment on MovieLens.
> Compare two candidate algorithms and report NDCG@10, Recall@10.
> !start
```

At runtime, AutoRecLab will:
1. derive concrete code requirements,
2. generate multiple candidate implementations,
3. execute and evaluate them,
4. debug/improve candidates iteratively,
5. stop when iteration budget/satisfaction criteria are reached.


## CLI Usage

**Inline prompt:**
```bash
uv run main.py --prompt "Analyze the signal and generate a report"
```

**Load prompt from an file:**
```bash
uv run main.py --prompt-file ./my-prompt.txt
```

**Don't log entered prompt in /entered_prompt.txt:**
```bash
uv run main.py --prompt-no-log
```

**Initialize workspace:**
```bash
uv run main.py --init
```

**List all available datasets:**
```bash
uv run main.py --list-datasets
```

**List all available models:**
```bash
uv run main.py --list-models
```

**Set model:**
```bash
uv run main.py --model "gpt-4o"
```

**Use a Codex model for code generation:**
```bash
uv run main.py --model "gpt-5.3-codex" --reasoning-effort high
```
When the selected model is a Codex model, AutoRecLab uses it **only** for code
generation and runs all non-coding calls (requirements, scoring, dataset
selection, summary) on `agent.codex_noncode_model` (default `gpt-5.4`). For any
non-Codex model, the same model is used for every call. `--reasoning-effort`
(`low|medium|high|xhigh`) applies to the code-generation step on
reasoning/Codex models; it is ignored for non-reasoning models, and Codex does
not support `minimal`/`none`.

**Run the same prompt multiple times:**
```bash
uv run main.py --prompt "Build a recommender..." --runs 10
```
Each run is stored in its own numbered subfolder inside the out directory
(`out/run_01`, `out/run_02`, ... `out/run_10`). Re-running continues the
numbering from the highest existing `run_*` folder, so previous results are
never overwritten. With `--runs 1` (the default) the output is written directly
to `out/` as before.

## Embeddings / documentation index

AutoRecLab uses FAISS vector stores in `ragEmbeddings/` for docs-aware coding.

Generate/update manually:

```bash
uv run python -m cli.embeddings.main generate --all
```

Useful flags:
- `--omnirec`, `--lenskit`, `--recbole` (select subset)
- `-f` / `--force` (overwrite existing index)
- `-o` (custom output directory)

## Configuration

Main config file: `config.toml`

Example (current defaults in this branch):

```toml
out_dir = "./out"

[treesearch]
num_draft_nodes = 3
debug_prob = 0.3
epsilon = 0.4
max_iterations = 5

[exec]
timeout = 5400
enable_type_checking = true
max_type_check_attempts = 3
keep_only_relevant_files = false

[agent]
k_fold_validation = 1
# Codex-only split (used only when agent.code.model is a Codex model):
codex_noncode_model = "gpt-5.4"
codex_noncode_reasoning_effort = "medium"

[agent.code]
model = "gpt-5.4-mini"
model_temp = 1.0
# Reasoning effort for the code-generation step on reasoning/Codex models:
reasoning_effort = "high"
```

Notes:
- `model_temp` only applies to non-reasoning models. Reasoning/Codex models
  (gpt-5.x, o-series, `*codex*`) reject non-default sampling parameters, so
  temperature is omitted for them and quality is steered via `reasoning_effort`.
- `reasoning_effort = "high"` increases quality but also cost and latency. Lower
  it (e.g. `medium`) to reduce spend, especially on the default model.

Environment override pattern:
- Prefix: `ARL_`
- Nested fields via `__`

Examples:
- `ARL_out_dir=./sandbox`
- `ARL_treesearch__max_iterations=8`
- `ARL_agent__code__model=gpt-5.3-codex`
- `ARL_agent__code__reasoning_effort=high`
- `ARL_agent__codex_noncode_model=gpt-5.4`

Logging level can be set via:
- `ISGSA_LOG=DEBUG|INFO|WARNING|ERROR`

Experiments with large datasets on limited disk space:
- `keep_only_relevant_files=false`: All output generated by AutoRecLab per node is saved and logged
- `keep_only_relevant_files=true`: Only the actual AutoRecLab output (in the form of code and results) is saved. Files such as saved trained models are deleted.

## Outputs

Depending on your `out_dir`, AutoRecLab writes artifacts such as:

- `code_requirements.json` (engineered requirements)
- `save.pkl` (tree state)
- intermediate generated code/checkpoints/plots/metadata
- execution logs (if you use shell redirection or helper scripts)

Utility for visualizing saved tree state:

```bash
uv run viz.py -i ./out/save.pkl -o ./out/tree_render
```

## Development workflow

Install development dependencies via `uv sync`, then run tests:

```bash
uv run pytest
```

Project includes:
- unit tests under `tests/`
- tree search core under `treesearch/`
- embedding CLI under `cli/embeddings/`
- utility package workspace member under `packages/dataloader/`

## Repository structure (high level)

```text
.
├── main.py                    # Entry point
├── config.toml                # Runtime config
├── compose.yaml               # Docker sandbox service
├── cli/embeddings/            # Embedding index tooling
├── treesearch/                # Core agent/search/execution logic
├── ragEmbeddings/             # FAISS indices for docs retrieval
├── sandbox/                   # Sandbox outputs/workspace
├── tests/                     # Tests
└── viz.py                     # Tree rendering utility
```

## Known develop-branch caveats

- Behavior and prompt contracts can change without notice.
- Some experimental backend/model combinations may be incomplete.

---

If you need stable, citable behavior for publication artifacts, use [`main`](../../tree/main).
For active feature development and newest research tooling, stay on `develop`.
