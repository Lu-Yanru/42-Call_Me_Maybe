*This project has been created as part of the 42 curriculum by yanlu.*

# Call Me Maybe
## Description
This project implements function calling in Large Language Models (LLMs) by building a system that translates natural language prompts into structured function calls with typed arguments.

Upon receiving a user prompt, instead of letting the LLM answer in natural language text, the system identifies which function to call (from a list of function definitions), extracts the appropriate arguments, and returns the function name and arguments in a JSON file.

For example:

	# Function definition:
	[
		{
			"name": "fn_add_numbers",
			"description": "Add two numbers together and return their sum.",
			"parameters": {
			"a": {
				"type": "number"
			},
			"b": {
				"type": "number"
			}
			},
			"returns": {
			"type": "number"
			}
		},
		...
	]

	# User prompt:
	"What is the sum of 40 and 2?"

	# Traditional LLM answer:
	"The sum of 40 and 2 is 42."

	# Function calling system answer:
	{
		"function": "fn_add_numbers",
		"arguments": {"a": 40, "b": 2}
	}

This technology is useful, as it serves as the basis of AI agents and enables LLMs to:

- Interact with external systems: Call APIs, query databases, control devices
- Execute code: Perform calculations, data transformations, file operations
- Chain operations: Break complex tasks into executable steps
- Provide structured output: Generate JSON, XML, or other machine-readable formats
- Extract structured data from unstructured text: For example, given a large book, extract fields such as {protagonist name, protagonist sex, protagonist age}


### Challenges
Small language models such as Qwen (0.6B parameters) used by default by this project are notoriously unreliable at generating structured output (~30% success rate). The current project achieves 100% reliability in generating the correct JSON format with the same small models using **constrained decoding**.

### Algorithm (constrained decoding)
The LLM generation process follows the following pipeline:

- Step 1: **Tokenization**. Process the prompt into subword units (token) and convert them into numerical IDs that can be processed by the model.
- Step 2: **LLM processing**. The model processes the numerical IDs through its neural network, and outputs probability scores (logits) for each possible token.
- Step 3: **Token selection through constrained decoding**.
Select the token with the highest logit to generate.
Before the selection, use constrained decoding to modify the logits and set the logits for invalid tokens to negative infinity. This guarantees that only valid tokens will be selected.
- Step 4: Repeat step 1 - 3 until the complete response is generated.

### Design decisions
key choices in implementation

Coalescence

### Performance analysis
| Matric | Target | Result |
|--------|--------|--------|
| Valid JSON format | 100% | |
| Correct function selection | > 90% | |
| Correct argument extraction | > 90% | |
| Processing speed | < 5 min | |

### Testing strategy
How to validate the implementation

#### Handled errors
- Malformed inputs
- Missing files

#### Handled edge cases
- Empty prompt
- Numbers: negative, float, zero, large numbers
- Special characters e.g. "café", "María"
- Strings with internal apostrophes e.g. "I'm", "don't"
- Semantic replacements e.g. "asterisks" → "*"
- Wrong types
- Ambiguous prompts
- Functions with multiple parameters

## Instructions
This project uses `uv` for dependency management and a `Makefile` to automate common tasks.

### Prerequisites
- Python 3.10+
- uv 0.9.16+

### Running the project

Install project dependencies:

	make install

Execute the project:

	make run

	# Or to specify custom input and output files:
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

Run the script in debug mode using pdb:

	make debug

Code linting using `flake8` and `mypy`:

	make lint

Remove temporary files andcaches:

	make clean

### Input format
#### Function definitions (`functions_definition.json`)
The available functions the system can call. Each function includes function name, argument names and types, return type and description.

	[
		{
			"name": "fn_add_numbers",
			"description": "Add two numbers together and return their sum.",
			"parameters": {
			"a": {
				"type": "number"
			},
			"b": {
				"type": "number"
			}
			},
			"returns": {
				"type": "number"
			}
		},
		{
			"name": "fn_greet",
			"description": "Greet the given name.",
			"parameters": {
			"name": {
				"type": "string"
			}
			},
			"returns": {
				"type": "string"
			}
		},
		...
	]

#### Test prompts (`function_calling_tests.json`)
A JSON array of natural language prompts.

	[
		{"prompt": "What is the sum of 2 and 3?"},
		{"prompt": "Greet john"},
		...
	]

### Output format
An array of JSON objects. Each object represents the result of each prompt and contains the prompt, the name of the function to call, and all required parameters.

	[
		{
			"prompt": "What is the sum of 2 and 3?",
			"name": "fn_add_numbers",
			"parameters": {"a": 2.0, "b": 3.0}
		},
		...
	]

## Resources
- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen documentation](https://qwen.readthedocs.io/en/latest/index.htmls)
- [Managing Python Projects With uv: An All-in-One Solution](https://realpython.com/python-uv/)
- [Python UV: The Ultimate Guide to the Fastest Python Package Manager](https://www.datacamp.com/tutorial/python-uv)
- [Build command-line interfaces with Python's argparse](https://realpython.com/command-line-interfaces-python-argparse/)
- [Working With JSON Data in Python](https://realpython.com/python-json/)
- [Testing and debugging regular expressions](https://regex101.com/)
- [pytest Tutorial: Effective Python Testing](https://realpython.com/pytest-python-testing/)
- [Python's unittest: Writing Unit Tests for Your Code](https://realpython.com/python-unittest/)
