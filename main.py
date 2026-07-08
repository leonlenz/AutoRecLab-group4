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
    out_dir = mkdir(config.out_dir)
    args = get_args()

    #Init workspace
    if args.init:
        mkdir(out_dir / "workspace")
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


    # Prepare to run AutoRecLab
    attach_file_handler(out_dir)
    cost_tracker.set_out_dir(out_dir)
    statistics_tracker.set_out_dir(out_dir)
    require_executable("dot")


    # Get user request
    user_request = None

    if args.prompt is not None:
        user_request = args.prompt

    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            user_request = f.read().strip()

    else:
        user_req_lines: list[str] = []
        print('Enter you request, write "!start" to start:')
        while True:
            line = input("> ")
            if line.lower().strip().startswith("!start"):
                break
            user_req_lines.append(line)

        user_request = "\n".join(user_req_lines)

    if user_request is None or user_request.strip() == "":
        logger.error("No request provided. Please provide a prompt using --prompt or --prompt-file, or type it manually.")
        return
    

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

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
