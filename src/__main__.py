from src.parse_args import parse_args
from src.parse_funcs import parse_funcs, FuncDefError
from src.parse_prompts import parse_prompts, PromptError


def main() -> None:
    # parse CLI
    args = parse_args()

    try:
        # parse functions definitions
        funcs = parse_funcs(args.functions_definition)
        print(funcs)
        # parse prompt
        prompts = parse_prompts(args.input)
        print(prompts)
        # process function definitions and save as valid tokens
        # process prompts
        # constrained decoding
        # write output
    except (FuncDefError, PromptError) as e:
        print(e)


if __name__ == "__main__":
    main()
