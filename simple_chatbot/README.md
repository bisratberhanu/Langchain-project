# Simple Chat Bot

## Description
The simple chat bot is built using LangChain and LLaMA. It is developed in a Jupyter Notebook and runs the model locally. This bot leverages the capabilities of the LLaMA model to generate responses to user queries.

## Files
- **simple_langchain.ipynb**: The Jupyter Notebook containing the code for the chat bot.
- **README.md**: Documentation for the simple chat bot.

## Installation
To run the notebook, you need to install the required packages. The following packages are used:

- `torch==2.0.1`
- `transformers==4.33.2`
- `langchain==0.0.299`
- `chromadb==0.4.10`
- `xformers==0.0.21`
- `sentence_transformers==2.2.2`
- `tokenizers==0.14.0`
- `optimum==1.13.1`
- `auto-gptq==0.4.2`
- `unstructured==0.10.16`

### Usage
**Load the Tokenizer**: The tokenizer is loaded from the pre-trained TheBloke/Llama-2-13b-Chat-GPTQ model, enabling text input to be converted into tokenized format for processing.

**Load the Model**: The GPTQ (quantized) version of Llama-2-13B is loaded, configured to run efficiently on compatible hardware with torch.float16 and automatic device mapping.

**Set Generation Parameters**: The GenerationConfig defines output behavior, including the number of tokens, randomness (temperature), sampling, and penalties for repetitive outputs.

**Pipeline and LangChain Integration**: A text generation pipeline is created and wrapped into a LangChain-compatible HuggingFacePipeline for easy integration into language model workflows.

Generate Responses: The model generates responses to user queries. The user can input text, and the model will generate a response based on the input.
