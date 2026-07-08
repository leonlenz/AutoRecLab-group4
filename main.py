import asyncio
import os
from argparse import ArgumentParser
from config import get_config
from treesearch.search import TreeSearch
from utils.log import _ROOT_LOGGER, attach_file_handler, set_log_level
from utils.path import mkdir
from utils.checks import require_executable
from treesearch.utils.costs_tracker import get_cost_tracker, get_model_table
from utils.statistics_tracker import get_statistics_tracker
from treesearch.utils.available_datasets import get_datasets_table

logger = _ROOT_LOGGER.getChild("main")
cost_tracker = get_cost_tracker()
statistics_tracker = get_statistics_tracker()


async def main():
    set_log_level(os.getenv("ISGSA_LOG", "INFO"))

    config = get_config()
    base_out_dir = mkdir(config.out_dir)
    args = get_args()

    #Init workspace
    if args.init:
        mkdir(base_out_dir / "workspace")
        return


    # List available datasets
    if args.list_datasets:
        datasets_table = get_datasets_table()
        print(datasets_table)
        return


    # List available models
    if args.list_models:
        models_table = get_model_table()
        print(models_table)
        return
    
    # Set model in config if provided as argument
    if args.model is not None:
        config.agent.code = config.agent.code.model_copy(update={"model": args.model})

    # Override reasoning effort for the code-generation step if provided
    if args.reasoning_effort is not None:
        config.agent.code = config.agent.code.model_copy(
            update={"reasoning_effort": args.reasoning_effort}
        )


    # Get user request (read once, reused for every run)
    user_request = get_user_request(args)

    if user_request is None or user_request.strip() == "":
        logger.error("No request provided. Please provide a prompt using --prompt or --prompt-file, or type it manually.")
        return


    # Validate the number of runs
    num_runs = args.runs
    if num_runs < 1:
        logger.error("--runs must be a positive integer (got %s).", num_runs)
        return


    # Run AutoRecLab once or multiple times with the same prompt
    if num_runs == 1:
        await run_once(config, base_out_dir, user_request, args)
    else:
        start_index = next_run_index(base_out_dir)
        pad_width = max(2, len(str(start_index + num_runs - 1)))

        for offset in range(num_runs):
            run_number = start_index + offset
            run_dir = mkdir(base_out_dir / f"run_{run_number:0{pad_width}d}")

            logger.info(
                f"===== Starting run {offset + 1}/{num_runs} "
                f"(out dir: {run_dir}) ====="
            )

            # Each run gets its own out_dir and a fresh tracker state
            config.out_dir = str(run_dir)
            await run_once(config, run_dir, user_request, args)

        logger.info(f"Finished all {num_runs} runs in {base_out_dir}")


def get_user_request(args) -> str | None:
    if args.prompt is not None:
        return args.prompt

    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    user_req_lines: list[str] = []
    print('Enter you request, write "!start" to start:')
    while True:
        line = input("> ")
        if line.lower().strip().startswith("!start"):
            break
        user_req_lines.append(line)

    return "\n".join(user_req_lines)


def next_run_index(base_out_dir) -> int:
    """Return the next available run number based on existing run_* folders."""
    max_index = 0
    for entry in base_out_dir.glob("run_*"):
        if not entry.is_dir():
            continue
        suffix = entry.name[len("run_"):]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


async def run_once(config, out_dir, user_request: str, args):
    # Start each run from a clean tracker state
    cost_tracker.reset()
    statistics_tracker.reset()

    # Prepare to run AutoRecLab
    attach_file_handler(out_dir)
    cost_tracker.set_out_dir(out_dir)
    statistics_tracker.set_out_dir(out_dir)
    require_executable("dot")

    # Log the user request
    if not args.prompt_no_log:
        prompt_file = out_dir / "entered_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(user_request)

    # Start AutoRecLab
    logger.info("Starting AutoRecLab...")
    logger.debug(f"User request:\n{user_request}")
    ts = TreeSearch(user_request, config=config)
    await ts._async_init()
    await ts.run()

    # Summarize results
    cost_tracker.saveSummarized()
    statistics_tracker.summarize_statistics()


def get_args():
    parser = ArgumentParser("AutoRecLab")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--prompt-no-log", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort for the code-generation step (reasoning/Codex models only).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How often to run the program with the same prompt. "
        "Each run is stored in its own numbered subfolder (run_001, run_002, ...) "
        "inside the out directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
