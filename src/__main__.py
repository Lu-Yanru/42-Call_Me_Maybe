from dotenv import load_dotenv
import os
import sys
from timeit import default_timer as timer


from src.parse_args import parse_args
from src.parse_funcs import parse_funcs, FuncDefError
from src.parse_prompts import parse_prompts, PromptError
from src.process_prompt import PromptProcessor, ModelError
from src.write_output import write_output


def main() -> None:
    # parse CLI
    args = parse_args()

    # load HF token
    if os.path.exists(".env"):
        load_dotenv()

    start = timer()
    try:
        # parse functions definitions
        funcs = parse_funcs(args.functions_definition)
        # parse prompt
        prompts = parse_prompts(args.input)
        # process prompts
        processor = PromptProcessor(funcs, prompts, args.model)
        processor.process()
        # write output
        write_output(processor.output, args.output)
    except (FuncDefError, PromptError) as e:
        print(e, file=sys.stderr)
    except ModelError as e:
        print(e, file=sys.stderr)
    except OSError as e:
        print(e, file=sys.stderr)
    finally:
        end = timer()
        print(f"Execution time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
