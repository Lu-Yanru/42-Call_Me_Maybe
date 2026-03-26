from src.parse_args import parse_args
from src.parse_funcs import parse_funcs, FuncDefError
from src.parse_prompts import parse_prompts, PromptError
from src.process_prompt import PromptProcessor
from src.write_output import write_output


def main() -> None:
    # parse CLI
    args = parse_args()

    try:
        # parse functions definitions
        funcs = parse_funcs(args.functions_definition)
        # parse prompt
        prompts = parse_prompts(args.input)
        # process function definitions and save as valid tokens
        # process prompts
        processor = PromptProcessor(funcs, prompts, args.model)
        processor.process()
        # constrained decoding
        # write output
        write_output(processor.output, args.output)
    except (FuncDefError, PromptError) as e:
        print(e)
    except OSError as e:
        print(e)


if __name__ == "__main__":
    main()
