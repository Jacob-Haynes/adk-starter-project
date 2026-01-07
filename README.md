# My Agent - ADK Starter Template

A production-ready starter template for building AI agents with Google's Agent Development Kit (ADK). Designed for quick cloning, customization, and deployment via IaC pipeline.

## Overview

This template provides a complete, best-practices structure for building sophisticated AI agents with ADK. It follows the architecture patterns established in production deployments and includes everything you need to get started quickly.

**Built with:**
- [Google ADK](https://github.com/google/adk-python) - Agent Development Kit
- Gemini 2.5 Flash for cost-effective agent reasoning
- Python 3.13+ with async/await
- Deployed to Google Cloud via IaC-managed Vertex AI Agent Engine

## Key Features

- **Production-Ready Structure**: Follows ADK best practices and proven patterns
- **Separation of Concerns**: Clean separation between agent logic, tools, prompts, and auth
- **Fast Development**: Hot-reload with `adk web` for instant testing
- **IaC Deployment**: Automatic deployment via IaC pipeline to Vertex AI
- **Example Tool**: Working example tool to demonstrate the pattern
- **Comprehensive Documentation**: Includes CLAUDE.md with ADK best practices guide
- **Testing Ready**: Test structure in place for adding pytest tests
- **Modern Python**: Python 3.13+, uv package manager, type hints

## Architecture

```
my_agent/
├── __init__.py          # Package entry point (ADK auto-discovery)
├── agent.py             # Root agent definition with Agent class
├── prompts.py           # AI instruction templates
├── tools.py             # Tool definitions (with example)
├── auth.py              # Authentication/API layer (placeholder)
├── requirements.txt     # Dependencies for Vertex AI deployment
├── sub_agents/          # Directory for multi-agent systems
│   └── __init__.py
└── tests/               # Test directory
    └── __init__.py
```

### File Purposes

- **agent.py**: Defines the root agent with model, instructions, tools, and callbacks
- **prompts.py**: Stores instruction templates - separate from code for easy iteration
- **tools.py**: All agent tools - functions the agent can call to perform operations
- **auth.py**: Authentication logic, API clients, credential management
- **requirements.txt**: Minimal dependencies for Vertex AI Agent Engine deployment

## Prerequisites

- **Python 3.13+** (3.11+ supported, 3.13 recommended for best performance)
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))
- **Google Cloud Project** with Vertex AI API enabled
- **gcloud CLI** configured and authenticated

## Installation

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/adk-starter-project.git
cd adk-starter-project

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Agent Configuration
AGENT_MODEL=gemini-2.5-flash
AGENT_TEMPERATURE=0.3
LOG_LEVEL=INFO
ENABLE_CLOUD_LOGGING=false

# Add your custom environment variables for tools
# API_KEY=your-api-key
# OAUTH_CLIENT_ID=your-oauth-client-id
# OAUTH_CLIENT_SECRET=your-oauth-client-secret
```

### 4. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

## Development

### Run Locally with Hot Reload

```bash
adk web my_agent --reload_agents
```

This launches an interactive web UI at `http://localhost:8000` where you can:
- Chat with your agent
- Test tools in real-time
- See agent reasoning and tool calls
- Debug quickly with automatic reload on file changes

### Run Tests

```bash
pytest my_agent/tests/ --cov=my_agent
```

### Code Quality

```bash
# Format code
ruff format

# Lint code
ruff check --fix
```

## Deployment

This project is designed to deploy via IaC (Infrastructure as Code) pipeline. The IaC system automatically:

1. **Generates** `cloudbuild.yaml` based on project structure
2. **Configures** Cloud Build triggers for automatic deployment
3. **Deploys** to Vertex AI Agent Engine
4. **Manages** OAuth configurations and environment variables

### Developer Responsibilities

- Ensure `my_agent/requirements.txt` contains all necessary dependencies
- Follow the standard ADK agent structure (this template does this)
- Configure environment variables in the IaC system as needed

**No manual cloudbuild.yaml creation required** - the IaC pipeline handles this automatically.

For detailed deployment instructions, refer to your organization's IaC documentation.

## Customization

### Adding Tools

Add new tools to `my_agent/tools.py`:

```python
async def my_custom_tool(
    tool_context: ToolContext | None,
    param: str,
) -> dict[str, Any]:
    """Description of what your tool does.

    Args:
        tool_context: The tool context from the agent.
        param: Description of parameter.

    Returns:
        dict: Description of return value.
    """
    # Your implementation here
    result = perform_operation(param)

    return {
        "status": "success",
        "data": result,
    }
```

Then add it to the agent's tools list in `my_agent/agent.py`:

```python
from .tools import example_tool, my_custom_tool

root_agent = Agent(
    # ...
    tools=[example_tool, my_custom_tool],
)
```

### Modifying Prompts

Edit `my_agent/prompts.py` to customize agent behavior:

```python
def return_agent_instructions() -> str:
    return """
    You are a specialized assistant for [your use case].

    Your capabilities:
    - [Capability 1]
    - [Capability 2]

    Guidelines:
    - [Guideline 1]
    - [Guideline 2]
    """
```

### Adding Authentication

Implement auth logic in `my_agent/auth.py`:

```python
class APIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Your auth implementation

    async def make_request(self, endpoint: str):
        # Make authenticated requests
        pass
```

### Creating Sub-Agents

For multi-agent systems, create specialized agents in `my_agent/sub_agents/`:

```
my_agent/sub_agents/
└── specialist_agent/
    ├── __init__.py
    └── agent.py
```

Then import and use as tools in the main agent.

## Renaming the Agent

To rename from `my_agent` to your agent name:

1. Rename the `my_agent/` directory to your agent name (e.g., `sales_agent/`)
2. Update `[tool.hatch.build.targets.wheel]` in `pyproject.toml`:
   ```toml
   packages = ["sales_agent"]
   ```
3. Update imports in your code
4. Update documentation references

## Documentation

- **CLAUDE.md** - Comprehensive ADK development guide with best practices, patterns, and examples
- **[ADK Documentation](https://google.github.io/adk-docs/)** - Official ADK docs
- **[Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)** - Google Cloud Vertex AI
- **[Gemini API Documentation](https://ai.google.dev/docs)** - Gemini model docs

## Next Steps

1. Customize the agent instructions in `my_agent/prompts.py`
2. Add your tools to `my_agent/tools.py`
3. Implement authentication if needed in `my_agent/auth.py`
4. Test locally with `adk web my_agent --reload_agents`
5. Deploy via your organization's IaC pipeline

## Support

For detailed development guidance, patterns, and best practices, see **CLAUDE.md** in this repository.

For ADK-specific questions, refer to the [official ADK documentation](https://google.github.io/adk-docs/).
