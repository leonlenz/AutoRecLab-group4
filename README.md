> [!IMPORTANT]
> **Looking for the latest features?** This branch contains the current stable release aka. **AutoRecLab v1**. 
> For active development and upcoming changes, please switch to the [`develop`](../../tree/develop) branch.

# Initial commits
> Pete was here !!!

# AutoRecLab v1: Towards an Autonomous Recommender-Systems Researcher

An autonomous AI agent that uses tree search to iteratively develop, debug, and improve code implementations for recommender systems research tasks.

## Overview

This agent automates the research code development process by:
- Generating multiple initial implementations from your research task description
- Executing and scoring them based on automatically generated requirements
- Using tree search to explore improvements and debug failures
- Iteratively refining code until finding a satisfactory solution

## Quick Start

### Installation & Setup

**Option 1: Docker (Recommended)**

For complete isolation with all dependencies pre-configured:

```bash
# Create environment file
echo "OPENAI_API_KEY=your-key-here" > .env

# Create workspace for datasets
mkdir -p sandbox/workspace
# cp your-dataset.csv sandbox/workspace/

# Build and run
docker compose run --build sandbox

# Inside container, run:
uv run main.py
```

**Option 2: UV**

UV provides the fastest and most reliable dependency management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Set API key and run
export OPENAI_API_KEY="your-key-here"
uv run main.py
```

**Option 3: pip**

```bash
pip install -e .
export OPENAI_API_KEY="your-key-here"
python main.py
```

### Usage

Run the agent and enter your research task description. The prompt supports multi-line input - type `!start` when ready:

```
Enter you request, write "!start" to start:
> Implement a collaborative filtering recommender using matrix factorization
> on the MovieLens dataset. Evaluate with RMSE and MAE metrics.
> !start
```

**What happens next:**
1. The agent generates 3 initial implementations (3 is default, configurable)
2. Each implementation is executed and scored
3. The agent uses tree search to iteratively improve the best solutions
4. Process continues until a satisfactory solution is found or max iterations reached
5. Final results and summary are presented

## Configuration

### Basic Settings

Edit `config.toml` to customize agent behavior:

```toml
[treesearch]
num_draft_nodes = 3      # Number of initial implementations to generate
max_iterations = 10      # Maximum improvement iterations
debug_prob = 0.3         # Probability of debugging vs improving (0.0-1.0)
epsilon = 0.3            # Exploration vs exploitation rate

[exec]
timeout = 3600           # Execution timeout in seconds
workspace = "./workspace"

[agent]
k_fold_validation = 1    # Preferred number of folds for validation (1 = no CV)

[agent.code]
model = "gpt-5-mini"     # LLM model to use
model_temp = 1.0         # Temperature for code generation
```

### Using Datasets

Place your datasets in the appropriate directory before running the agent:

**Local execution (UV/pip):**
- Directory: `./workspace/`
- Example: `cp movielens.csv ./workspace/`
- Reference in prompt: `"Load data from movielens.csv in the working directory"`

**Docker execution:**
- Directory: `./sandbox/workspace/`
- Example: `cp movielens.csv ./sandbox/workspace/`
- Reference in prompt: `"Load data from movielens.csv in the working directory"`

### Available Libraries

The agent can use these packages (defined in `pyproject.toml`):
- `numpy==1.26.4`, `numba==0.58.1`, `pandas==2.3.2`
- `scipy==1.16.2`, `scikit-learn==1.7.1`, `lenskit==0.14.4`

## Customization

### Adding Custom Libraries

To add new Python packages that the agent can use:

1. **Add dependency:**
   ```bash
   uv add package-name  # with UV
   # or edit pyproject.toml manually
   ```

2. **Update agent awareness** in `treesearch/minimal_agent.py`:
   ```python
   pkgs = [
       "numpy==1.26.4",
       "your-package==2.0.0",  # Add your package here
       # ...
   ]
   ```

3. **Resync environment:**
   ```bash
   uv sync  # for local
   docker compose run --build  # for Docker
   ```

### Modifying Agent Behavior

- **Prompts & code generation:** `treesearch/minimal_agent.py`
- **Tree search strategy:** `treesearch/search.py`
- **Execution settings:** `config.toml`

## How It Works

The agent uses a tree search approach to iteratively improve code:

1. **Task Analysis**: Automatically generates specific code requirements from your research task description
2. **Draft Generation**: Creates multiple initial implementations using LLM (default: 3)
3. **Execution & Scoring**: Runs each implementation and scores based on:
   - Execution success (whether code runs without bugs)
   - Requirement fulfillment (0-100% based on auto-generated requirements)
   - Code quality and conceptual correctness
4. **Tree Search**: Uses epsilon-greedy strategy to select nodes for expansion:
   - Prioritizes debugging buggy implementations
   - Improves working implementations to increase scores
   - Balances exploration of new solutions vs exploitation of best solutions
5. **Termination**: Stops when a satisfactory solution is found (high score, all requirements met) or maximum iterations reached

## Output Files

Results are automatically saved to:
- `./workspace/` (local) or `./sandbox/` (Docker) - Execution workspace with generated code and results
- `./out/save.pkl` - Pickled tree search state (all nodes and their scores)
- `./out/code_requirements.json` - Auto-generated code requirements for the task

## Requirements

- Python ≥3.11
- OpenAI API key (**Note:** Anthropic/Claude API not yet implemented)
- UV (recommended), pip, or Docker
- Packages listed in `pyproject.toml`

## Project Structure

```
├── main.py              # Entry point
├── config.toml          # Configuration file
├── pyproject.toml       # Dependencies
├── workspace/           # Local execution workspace
├── sandbox/workspace/   # Docker execution workspace
├── treesearch/
│   ├── minimal_agent.py # Agent logic and prompts
│   ├── search.py        # Tree search algorithm
│   ├── node.py          # Node representation
│   └── backend/         # LLM interfaces
└── utils/               # Logging utilities
```

## Example
Below is an exemplary run of *AutoRecLab*. We entered the following prompt:
```markdown
AutoRecLab $ uv run main.py
> I'd like to run an experiment to quantify how much random seeds affect recommender system results.
> Please use a simple ImplicitMF algorithm from LensKit, perform a train/test split on the MovieLens100K 'u.data',
> train it multiple times with identical settings but different random seeds,
> and analyze how metrics like NDCG or Recall vary across runs to find out if reporting seeds is important.
> I placed the 'u.data' file of MovieLens100K in your current working directory. You can load it from there!
> !start
```

The Planner component of *AutoRecLab* then engineered the following requirements:

```markdown
1. Load the MovieLens100K 'u.data' file from the current working directory and parse it into a dataframe with columns [user, item, rating, timestamp]. Fail with a clear error if the file is missing or malformed.
2. Convert the explicit ratings to an implicit interaction matrix suitable for LensKit ImplicitMF (e.g., binary interactions or confidence weights). Make the conversion method configurable (binary threshold or use rating as confidence).
3. Create a single, reproducible train/test split once and reuse it across all runs: use a user-wise leave-one-out or leave-k-out strategy that holds out at least one interaction per test user; ensure no test-item appears in the corresponding user\u2019s training set.
4. Exclude users with fewer interactions than required by the chosen split strategy (e.g., users with only one event if performing leave-one-out) and document this filtering.
5. Instantiate LensKit\u2019s ImplicitMF with fixed hyperparameters (factors, epochs, reg, learning rate, etc.) and ensure these hyperparameters are identical across all runs.
6. Run the training loop N times (N configurable, default >= 30) with identical hyperparameters but different random seeds. For each run, set the Python random seed and numpy.random seed, and pass the seed to any RNGs used by LensKit/underlying libraries so the only intended difference between runs is the random seed.
7. Ensure data order and any shuffling that affects training are driven by the run-specific seed so that each run truly differs only by seed.
8. After each run, produce top-K recommendations per user (K configurable, default 10), excluding items seen in that user\u2019s training data.
9. Compute per-run evaluation metrics: NDCG@K and Recall@K (per-user then mean), and store per-user metric arrays as well as run-level aggregates (mean, std, min, max) in a persistent CSV or similar.
10. Compute and report summary statistics across runs: mean, standard deviation, min, max, coefficient of variation, and 95% confidence intervals (e.g., bootstrap or t-based) for each metric.
11. Produce simple visualizations (histogram or boxplot) of metric distributions across runs and save plot files (PNG/SVG) for NDCG@K and Recall@K.
12. Log and save experiment metadata for reproducibility: list of seeds used, LensKit version, package versions (numpy, pandas, scipy), hyperparameters, train/test split statistics, and timestamp.
13. Provide an option to re-run the entire experiment deterministically given a fixed seed list and the saved train/test split (i.e., ability to reproduce a particular run by seed).
14. Implement robust error handling and input validation for parameters (n_runs, K, threshold, hyperparameters) with informative error messages.
15. Include a short summary output (console or saved text) that answers: how large is inter-run variability for each metric relative to its mean, and whether reporting the seed is likely important based on observed variability.
16. Make the script runnable from command line or notebook with configurable parameters (data path, n_runs, K, seed or seed range, hyperparameters) and document usage in a brief README or docstring.
```

The final report that *AutoRecLab* returned at the end:
```markdown
What was run (from the code)
- Data: MovieLens 100K, u.data, converted to implicit-confidence weights (weight = 1 + 0.5*(rating-1)).
- Split: leave-one-out per user; the split was fixed with split_seed = 42, so the train/test split is identical across runs.
- Model: internal SimpleImplicitMF (SGD-based implicit MF). Key settings: latent_dim = 20, lr = 0.01, reg = 0.1, epochs = 15.
- Randomness: each run used a different seed from 0..29 (n_runs = 30). The model uses a numpy RandomState seeded per run to initialize factors and to shuffle training order, so per-run randomness comes from initialization and SGD order.
- Evaluation: recall@10 and NDCG@10 on the held-out item per user (users with ≥2 interactions are evaluated; in this run that was all 943 users).

Key numeric findings (from the printed output)
- NDCG@10 across 30 seeds: mean = 0.0147, std = 0.0023, coefficient of variation (std/mean) ≈ 0.1597 (≈16%), 95% CI for the mean ≈ ±0.0008.
- Recall@10 across 30 seeds: mean = 0.0318, std = 0.0050, coefficient of variation ≈ 0.1578 (≈16%), 95% CI for the mean ≈ ±0.0018.
- The script printed a recommendation at the end: "report random seeds and run multiple seeds when evaluating recommender performance." The computed relative variability for NDCG@10 was 0.160.

Interpretation
- Absolute performance is low for this simple setup (mean NDCG ≈ 0.015, mean Recall@10 ≈ 0.032), but the important point for your question is variability across identical experiments that only differ by random seed.
- Relative variability across seeds is substantial: about 16% CV for both NDCG and Recall. That means seed-to-seed differences are large relative to the metric magnitudes and could change conclusions if only a single seed/run is reported.
- Because the train/test split was held constant, the observed variability is attributable to model randomness (initial factor values and SGD ordering), which the code explicitly uses the seed to control.

Practical recommendation (based on these results)
- Do not report a single-run metric without a seed. Report seeds and/or run multiple seeds.
- Report mean ± standard deviation (or mean ± 95% CI) across multiple seeds. In your experiment, using 30 seeds produced reasonably tight CIs (±0.0008 for NDCG mean here), so 30 runs is a good example if you can afford the computation.
- Also save the per-seed results and the seeds used (the experiment code already does this), so results are reproducible and readers can see run-level variability.

Limitations / caveats
- These conclusions apply to this particular simple implicit-MF implementation, data conversion (confidence), and hyperparameter choices. Different algorithms, preprocessing, or hyperparameters may show more or less sensitivity to seeds.
- The absolute metric values are specific to this model and setup; the actionable point is the relative variability across seeds (≈16%).
```
