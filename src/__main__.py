from src.parse_args import parse_args
from src.parse_funcs import parse_funcs, FuncDefError
from src.parse_prompts import parse_prompts, PromptError
from src.write_output import write_output


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
        write_output(funcs, args.output)
    except (FuncDefError, PromptError) as e:
        print(e)
    except OSError as e:
        print(e)


if __name__ == "__main__":
    main()
