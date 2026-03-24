*This project has been created as part of the 42 curriculum by yanlu.*

# Call Me Maybe
## Description
This project implements function calling in Large Language Models (LLMs) by building a system that translates natural language prompts into structured function calls with typed arguments.

Upon receiving a user prompt, instead of letting the LLM answer in natural language text, the system identifies which function to call (from a list of function definitions), extracts the appropriate arguments, and returns them in a JSON file.

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
Small language models such as `Qwen` (0.6B parameters) used by this project are notoriously unreliable at generating structured output (~30% success rate). The current project achieves 100% reliability with the same small models using **constrained decoding**.

### Algorithm (constrained decoding)
Constrained decoding is a technique that guides the model's output token-by-token to guarantee valid structure, without relying on prompting alone.

### Design decisions
key choices in implementation

### Performance analysis
accuracy, speed and reliability

### Testing strategy
How to validate the implementation

## Instruction

### Example usage

## Resources
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Managing Python Projects With uv: An All-in-One Solution](https://realpython.com/python-uv/)
- [Python UV: The Ultimate Guide to the Fastest Python Package Manager](https://www.datacamp.com/tutorial/python-uv)
- [Testing and debugging regular expressions](https://regex101.com/)
