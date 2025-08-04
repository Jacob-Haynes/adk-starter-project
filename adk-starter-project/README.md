# Google GenAI Agent Example

This repository demonstrates how to set up and run an LLM agent using Google GenAI and Vertex AI with Python.

## Features

- Uses Google GenAI and Vertex AI for LLM-based agents
- RAG (Retrieval-Augmented Generation) example for document retrieval and context injection
- Easily configurable via `.env` file
- Ready for extension with custom tools and output schemas

## Requirements

- Python >=3.9, <=3.12 (3.12 recommended, Agent engine deployment currently doesnt support 3.13)
- See `requirements.txt` for all packages I use - this is more than the minimum needed.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jacob-haynes-cat/adk-starter-project.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables:**
   - Copy `SRC/.env` and update the following values:
     - `GOOGLE_CLOUD_PROJECT`
     - `GOOGLE_CLOUD_LOCATION`
     - `GCP_MODEL`

4. **Authenticate with Google Cloud:**
   - Make sure your environment is authenticated to access Vertex AI.

## Usage

- The main agent is defined in `SRC/agent.py`.
- To launch the agent in a browser UI, run in root directory:
    ```bash
    adk web
    ```
  - To run the RAG example for document retrieval and querying, execute:
`SRC/rag_example.py`

## File Structure

- `SRC/agent.py` — Main agent definition and configuration
- `SRC/rag_example.py` — Example for RAG (Retrieval-Augmented Generation) setup and querying
- `SRC/.env` — Environment variables for configuration
- `requirements.txt` — Python dependencies

## Customization

- Modify the agent's prompt, tools, or output schema in `SRC/agent.py` as needed.
- Add sub agents in subdirectories of `SRC/agents/` for more complex workflows.
- Each sub agent will need its own '--init--.py' and 'agent.py' file to be recognized by the adk.

## RAG Example
The RAG example in `SRC/rag_example.py` demonstrates how to:

- Set up a RAG corpus using Google Vertex AI
- Import documents from Google Drive
- Query the corpus for relevant context using natural language
- To test the RAG engine, ensure your .env is configured and run:
    ```bash
    python SRC/rag_example.py
    ```
  This will set up the corpus, import files, and perform a sample query, printing the retrieved contexts.

## Useful Documentation

- [ADK Documentation](https://google.github.io/adk-docs/)
- [Google GenAI Documentation](https://cloud.google.com/genai/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Vertex AI RAG Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview)

