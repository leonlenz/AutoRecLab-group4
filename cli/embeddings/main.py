from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from cli.embeddings.preprocessor import GitRepoPreprocessor, Preprocessor


def main():
    load_dotenv()

    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="sub_command", required=True)

    generate = sub.add_parser("generate")

    generators: dict[str, Preprocessor] = {
        "--omnirec": GitRepoPreprocessor("https://github.com/ISG-Siegen/OmniRec"),
        "--lenskit": GitRepoPreprocessor(
            "https://github.com/lenskit/lkpy.git", "main"
        ),
        "--recbole": GitRepoPreprocessor(
            "https://github.com/RUCAIBox/RecBole.git", "1.2.x"
        ),
    }
    generator_destinations: dict[str, str] = {}
    for key in generators.keys():
        action = generate.add_argument(key, action="store_true")
        generator_destinations[action.dest] = key
    generate.add_argument("--all", action="store_true")
    generate.add_argument("--chunk-size", type=int, default=4000)
    generate.add_argument("--chunk-overlap", type=int, default=200)
    generate.add_argument(
        "--embedding-model", type=str, default="text-embedding-3-large"
    )
    generate.add_argument("-o", "--out", type=Path, default=Path("./ragEmbeddings"))
    generate.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing vector store"
    )

    args = parser.parse_args()

    if args.sub_command == "generate":
        if args.all:
            for dest in generator_destinations.keys():
                setattr(args, dest, True)

        embedding_model = OpenAIEmbeddings(model=args.embedding_model)
        for dest, key in generator_destinations.items():
            if getattr(args, dest):
                vector_store_pth: Path = args.out / dest
                if vector_store_pth.exists() and not args.force:
                    print(
                        f"Vector store '{vector_store_pth}' already exists, skipping {key}. Add '-f' to force recreation."
                    )
                    continue
                generator = generators[key]
                tmp_dir = TemporaryDirectory()

                print(f"Running {dest} with {generator.__class__.__name__}...")
                splits = generator.get_splits(
                    args.chunk_size, args.chunk_overlap, Path(tmp_dir.name)
                )

                vector_store = FAISS.from_documents(
                    documents=splits, embedding=embedding_model
                )
                print("Saving...")
                vector_store.save_local(str(vector_store_pth))
                tmp_dir.cleanup()
                print(f"Done! Saved vector store to '{vector_store_pth}'.")

    else:
        raise ValueError(f"Unknown subcommand: {args.sub_command}")


if __name__ == "__main__":
    main()