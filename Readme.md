# RAG

## Installation

We use [Poetry](https://python-poetry.org/) to manage dependencies.

```bash
poetry init
poetry install
```

Create a `.env` file in the root directory and add the following environment variables:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<LANGCHAIN_API_KEY>
```

> Note: You can get the `LANGCHAIN_API_KEY` from [LangSmith](https://www.langchain.com/langsmith).


Create `testdata` directory in the root directory and add files.


## Usage

Run ollama:
```bash
ollama serve
```

Open a new terminal and run the following command:
```bash
python main.py
```

Then you can ask questions to the model.
If you want to exit, type `bye`.
