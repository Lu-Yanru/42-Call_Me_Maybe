from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog="python -m src"
    )

    parser.add_argument(
        "-f", "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path of the functions defintion file."
        )

    parser.add_argument(
        "-i", "--input",
        default="data/input/function_calling_tests.json",
        help="Path of the file containing the prompts."
        )

    parser.add_argument(
        "-o", "--output",
        default="data/output/function_calling_results.json",
        help="Path of the output file."
        )

    parser.add_argument(
        "-m", "--model",
        default="Qwen/Qwen3-0.6B",
        help="Name of the LLM model."
        )

    return parser.parse_args()
