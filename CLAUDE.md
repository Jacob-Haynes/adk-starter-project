# AI Agent Development Guide

Comprehensive guide for building sophisticated AI agents with Google's Agent Development Kit (ADK).

---

## Using This Starter Template

This starter template provides a production-ready foundation for building ADK agents. Here's how to get started:

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd adk-starter-project
   uv venv && source .venv/bin/activate
   uv sync
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Google Cloud project settings
   ```

3. **Test Locally**
   ```bash
   adk web my_agent --reload_agents
   ```

### Customizing for Your Use Case

**Rename the Agent:**
1. Rename `my_agent/` directory to your agent name (e.g., `sales_agent/`)
2. Update `pyproject.toml`:
   ```toml
   packages = ["sales_agent"]  # Update this line
   ```
3. Update imports throughout your code

**Add Your Business Logic:**
- **Tools** (`my_agent/tools.py`): Add functions for your agent's capabilities
- **Prompts** (`my_agent/prompts.py`): Customize agent instructions and behavior
- **Auth** (`my_agent/auth.py`): Implement authentication for external APIs
- **Agent** (`my_agent/agent.py`): Configure model, temperature, and callbacks

**Key Principles:**
- Keep agents abstract and domain-agnostic
- Use RAG for domain knowledge, not hardcoded logic
- Separate tools (mechanics) from prompts (behavior)
- Follow the patterns in this guide

---

## Google ADK Overview

**What is ADK?**
An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and built for compatibility with other frameworks.

**Key Characteristics:**
- **Code-First Development**: Define agent logic directly in Python
- **Rich Tool Ecosystem**: Utilize pre-built and custom tools for agent capabilities
- **Modular Multi-Agent Systems**: Create scalable applications with specialized agents
- **Model-Agnostic**: Works with various LLMs, optimized for Gemini and Google ecosystem
- **Deploy Anywhere**: Containerize and deploy on Cloud Run, Vertex AI, or AgentSpace

**Installation:**
```bash
# Stable release
pip install google-adk

# Development version
pip install git+https://github.com/google/adk-python.git@main
```

### Key Components

-   **Agent** - Blueprint defining identity, instructions, and tools
    (`LlmAgent`, `LoopAgent`, `ParallelAgent`, `SequentialAgent`, etc.)
-   **Runner** - Execution engine that orchestrates agent execution. It manages
    the 'Reason-Act' loop, processes messages within a session, generates
    events, calls LLMs, executes tools, and handles multi-agent coordination.
-   **Tool** - Functions/capabilities agents can call (Python functions, OpenAPI
    specs, MCP tools, Google API tools)
-   **Session** - Conversation state management (in-memory, Vertex AI,
    Spanner-backed)
-   **Memory** - Long-term recall across sessions

---

## Agent Structure Convention (Required)

**All agent directories must follow this structure:**
```
my_agent/
├── __init__.py      # MUST contain: from . import agent
└── agent.py         # MUST define: root_agent = Agent(...) OR app = App(...)
```

**Choose one pattern based on your needs:**

### Option 1 - Simple Agent (for basic agents without plugins)

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Option 2 - App Pattern (when you need plugins, event compaction, custom configuration)

```python
from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins import ContextFilterPlugin

root_agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant.",
    tools=[...],
)

app = App(
    name="my_app",
    root_agent=root_agent,
    plugins=[
        ContextFilterPlugin(num_invocations_to_keep=3),
    ],
)
```

**When to use App pattern:**
- Need session event compaction (limit context window size)
- Want to add plugins (logging, filtering, custom middleware)
- Require complex configuration or initialization
- Need app-level state or resources

**Rationale:** This structure allows the ADK CLI (`adk web`, `adk run`, etc.) to automatically discover and load agents without additional configuration.

---

## How the Runner Works

The Runner is the stateless orchestration engine that manages agent execution. It does not hold conversation history in memory; instead, it relies on services like `SessionService`, `ArtifactService`, and `MemoryService` for persistence.

### Invocation Lifecycle

Each call to `runner.run_async()` or `runner.run()` processes a single user turn, known as an **invocation**.

1.  **Session Retrieval:** When `run_async()` is called with a `session_id`, the
    runner fetches the session state, including all conversation events, from
    the `SessionService`.
2.  **Context Creation:** It creates an `InvocationContext` containing the
    session, the new user message, and references to persistence services.
3.  **Agent Execution:** The runner calls `agent.run_async()` with this context.
    The agent then enters its reason-act loop, which may involve:
    *   Calling an LLM for reasoning.
    *   Executing tools (function calling).
    *   Generating text or audio responses.
    *   Transferring control to sub-agents.
4.  **Event Streaming & Persistence:** Each step in the agent's execution (LLM
    call, tool call, tool response, model response) generates `Event` objects.
    The runner streams these events back to the caller and simultaneously
    appends them to the session via `SessionService`.
5.  **Invocation Completion:** Once the agent has produced its final response
    for the turn (e.g., a text response to the user), the agent's execution loop
    finishes.
6.  **Event Compaction:** If event compaction is configured (via App plugins), the runner may
    summarize older events in the session to manage context window limits,
    appending a `CompactedEvent` to the session.
7.  **Next Turn:** When the user sends another message, a new `run_async()`
    invocation begins, repeating the cycle by loading the session, which now
    includes the events from all prior turns.

**Key Insight:** The Runner provides different execution modes:
- `run_async()` - Asynchronous execution for production
- `run_live()` - Bi-directional streaming interaction
- `run()` - Synchronous execution for local testing and debugging

---

## Design Principle: Context-Agnostic Agents

**Philosophy**: Keep agents abstract and domain-agnostic. Inject domain-specific knowledge dynamically through RAG, not hardcoded logic.

### Core Rules

- ✅ Abstract agents that work across domains
- ✅ RAG for schemas, examples, and business definitions
- ✅ Configuration for deployment-specific settings
- ✅ Prompt templates with variable injection
- ❌ No hardcoded table names, business logic, or domain assumptions
- ❌ No domain-specific instructions directly in system prompts
- ❌ No data definitions in Python code

### Example

```python
# ❌ Bad: Domain knowledge hardcoded
agent = Agent(
    name="sales_agent",
    instruction="You analyze the customers table with columns: id, name, email..."
)

# ✅ Good: Knowledge injected via RAG or other variable
agent = Agent(
    name="data_agent",
    instruction=f"""You analyze data from the provided schemas.

Available Schemas:
{rag_context}

Use these schemas to answer user questions."""
)
```

### Separation of Concerns

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Tools** | Mechanics: auth, execution, retrieval | Python functions with `@tool` decorator |
| **RAG** | Knowledge: schemas, examples, definitions | Vertex AI RAG corpus |
| **Prompts** | Behavior: reasoning, formatting, error handling | Template strings with variable injection |
| **Config** | Deployment: endpoints, limits, model selection | Environment variables |

---

## Prompt Engineering with Variable Injection

### Template Pattern

```python
def build_prompt(user_query: str, context: str) -> str:
    return f"""You are an expert assistant.

Relevant Context:
{context}

User Query: {user_query}

Instructions:
- Use the context above to inform your response
- Be concise and accurate
- Cite specific information when applicable
"""
```

### RAG Integration Pattern

```python
# Retrieve context from RAG
rag_results = query_rag_corpus(user_question)
context = "\n\n".join([doc.text for doc in rag_results])

# Inject into agent instruction or tool input
agent_instruction = build_prompt(user_question, context)
```

---

## Tool Development

### Creating Custom Tools

```python
from google.adk.tools import tool

@tool
def custom_data_query(query: str, limit: int = 100) -> str:
    """
    Execute a data query and return results.

    Args:
        query: The query to execute
        limit: Maximum number of results to return

    Returns:
        Query results formatted as a string
    """
    # Implementation here
    results = execute_query(query, limit)
    return format_results(results)
```

### Tool Best Practices

- Use descriptive names and comprehensive docstrings (LLM uses these to decide when to call)
- Include type hints for all parameters
- Return structured, parseable data
- Handle errors gracefully with informative messages
- Validate inputs and apply security constraints
- Log important operations for debugging

---

## Common Agent Patterns

### Pattern 1: RAG-Enhanced Specialist Agent

**Use when:** The agent needs deep domain knowledge that changes frequently.

```python
from google.adk.agents import Agent
from google.adk.tools import tool

@tool
def query_knowledge_base(question: str) -> str:
    """Retrieve relevant context from RAG corpus."""
    rag_results = vertex_rag_client.query(question, top_k=5)
    return "\n\n".join([doc.text for doc in rag_results])

specialist_agent = Agent(
    name="domain_specialist",
    model="gemini-2.5-pro",
    instruction="""You are a domain expert. For each question:
    1. Use the query_knowledge_base tool to get relevant context
    2. Synthesize information from the retrieved context
    3. Provide accurate, well-sourced answers""",
    tools=[query_knowledge_base, other_tools]
)
```

### Pattern 2: Orchestrator with Sub-Agents

**Use when:** Tasks require different specialized capabilities.

```python
# Specialized sub-agents
analyst = Agent(
    name="data_analyst",
    model="gemini-2.5-pro",
    instruction="You analyze data and generate insights...",
    tools=[query_tool, analyze_tool]
)

writer = Agent(
    name="report_writer",
    model="gemini-2.5-flash",
    instruction="You create clear, professional reports...",
    tools=[format_tool, visualize_tool]
)

# Orchestrator
orchestrator = Agent(
    name="coordinator",
    model="gemini-2.5-flash",
    instruction="""You coordinate between specialists:
    - Route data analysis tasks to data_analyst
    - Route report generation to report_writer
    - Combine results into comprehensive responses""",
    tools=[analyst, writer]
)
```

### Pattern 3: Tool-Heavy Execution Agent

**Use when:** The agent primarily executes operations vs. reasoning.

```python
execution_agent = Agent(
    name="executor",
    model="gemini-2.5-flash",  # Use flash for cost efficiency
    instruction="""You execute user requests using available tools.
    - Parse user intent
    - Select appropriate tool
    - Execute and return results
    - Minimal additional commentary""",
    tools=[tool1, tool2, tool3, tool4]
)
```

---

## RAG Integration Best Practices

### When to Use RAG

✅ **Use RAG for:**
- Database schemas and metadata
- Example queries and patterns
- Business definitions and glossaries
- Domain-specific knowledge that changes
- Data quality notes and gotchas
- Performance optimization guidelines

❌ **Don't use RAG for:**
- Hardcoded logic and algorithms
- Authentication and authorization
- Core agent behavior and reasoning patterns
- Small, static reference data (use config instead)

### RAG Corpus Organization

**Recommended Structure:**

1. **Schema Definitions** - Table DDL, column types, descriptions
2. **Example Queries** - NL question + query pairs with annotations
3. **Business Glossary** - Terms, metrics, KPIs, calculations
4. **Data Quality Notes** - Known issues, null handling, historical changes
5. **Relationship Patterns** - Join patterns, foreign keys, cardinality
6. **Performance Tips** - Partitioning, indexing, optimization

**Best Practices:**
- One document per concept for retrieval precision
- Consistent formatting and terminology
- Include metadata tags for filtering
- Version control for schema changes
- Keep synchronized with actual systems
- Test retrieval with common queries

### RAG Retrieval Pattern

```python
def get_rag_context(user_query: str, corpus_id: str, top_k: int = 10) -> str:
    """Retrieve relevant context from RAG corpus."""
    from google.cloud import aiplatform

    # Query RAG corpus
    rag_client = aiplatform.RagClient()
    results = rag_client.query(
        corpus=corpus_id,
        query=user_query,
        top_k=top_k,
        similarity_threshold=0.3
    )

    # Combine results
    context = "\n\n---\n\n".join([
        f"Source: {doc.source}\n{doc.text}"
        for doc in results.documents
    ])

    return context
```

---

## Model Selection Guidelines

### Gemini 2.5 Flash

- **Use for**: Fast, lightweight tasks, orchestration, simple reasoning
- **Cost**: Lower cost per token
- **Speed**: Faster response times
- **Best for**: Root agents, simple routing, quick lookups

### Gemini 2.5 Pro

- **Use for**: Complex reasoning, analysis, accuracy-critical tasks
- **Cost**: Higher cost per token
- **Capability**: Better at nuanced understanding, complex instructions
- **Best for**: Specialist agents, NL2SQL, deep analysis

### Optimization Strategy

```python
# Use Flash for orchestration (cheap, fast)
root_agent = Agent(
    model="gemini-2.5-flash",
    # ... routes to specialists
)

# Use Pro for complex reasoning (expensive, accurate)
analyst_agent = Agent(
    model="gemini-2.5-pro",
    # ... complex analysis tasks
)
```

---

## Environment Configuration

### Standard Environment Variables

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLOUD_STAGING_BUCKET=your-bucket

# AgentSpace Deployment (if applicable)
AGENTSPACE_PROJECT_NUMBER=123456789
AGENTSPACE_APP_ID=your-app-id
AGENTSPACE_LOCATION=us

# OAuth Authentication (if required for your tools)
OAUTH_CLIENT_ID=your-client-id
OAUTH_CLIENT_SECRET=your-secret
OAUTH_SCOPES=your-required-scopes

# Model Configuration
ROOT_AGENT_MODEL=gemini-2.5-flash
SPECIALIST_AGENT_MODEL=gemini-2.5-pro

# RAG Configuration (if using Vertex AI RAG)
VERTEX_RAG_CORPUS=projects/PROJECT/locations/LOC/ragCorpora/CORPUS_ID
VERTEX_RAG_SIMILARITY_TOP_K=10
VERTEX_RAG_VECTOR_DISTANCE_THRESHOLD=0.3

# Feature Flags
ENABLE_CACHING=true
DEBUG_MODE=false
```

### Configuration Best Practices

1. **Never commit secrets** - use `.env.example` as a template
2. **Use Secret Manager** - for production credentials
3. **Environment-specific configs** - separate `.env` files for dev/staging/prod
4. **Validation on startup** - check required vars are set
5. **Sensible defaults** - provide fallback values where appropriate

---

## Local Development & Testing

### Running Agents Locally

**For interactive development and debugging:**
```bash
# Launch web UI (recommended for development)
adk web path/to/agents_dir

# With hot reload
adk web --reload_agents
```

**For CLI-based testing:**
```bash
# Interactive CLI (prompts for user input)
adk run path/to/my_agent
```

**For API/production mode:**
```bash
# Start FastAPI server
adk api_server path/to/agents_dir
```

**For running evaluations:**
```bash
# Run evaluation set against agent
adk eval path/to/my_agent path/to/eval_set.json
```

### Development Setup

**Requirements:**
- Python 3.10+ (**Python 3.11+ strongly recommended** for best performance)
- `uv` package manager (**required** - faster than pip/venv)

**Install uv if not already installed:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Standard setup for development:**
```bash
# Create virtual environment with Python 3.11
uv venv --python "python3.11" ".venv"
source .venv/bin/activate

# Install all dependencies for development
uv sync --all-extras
```

**Building:**
```bash
# Build wheel
uv build

# Install local build for testing
pip install dist/google_adk-<version>-py3-none-any.whl
```

---

## Python Best Practices

### General Python Best Practices

*   **Constants:** Use immutable global constant collections (tuple, frozenset, immutabledict) to avoid hard-to-find bugs. Prefer constants over wild string/int literals, especially for dictionary keys, pathnames, and enums.
*   **Naming:** Name mappings like `value_by_key` to enhance readability in lookups (e.g., `item = item_by_id[id]`).
*   **Readability:** Use f-strings for concise string formatting, but use lazy-evaluated `%`-based templates for logging. Use `repr()` or `pprint.pformat()` for human-readable debug messages. Use `_` as a separator in numeric literals to improve readability.
*   **Comprehensions:** Use list, set, and dict comprehensions for building collections concisely.
*   **Iteration:** Iterate directly over containers without indices. Use `enumerate()` when you need the index, `dict.items()` for keys and values, and `zip()` for parallel iteration.
*   **Built-ins:** Leverage built-in functions like `all()`, `any()`, `reversed()`, `sum()`, etc., to write more concise and efficient code.
*   **Flattening Lists:** Use `itertools.chain.from_iterable()` to flatten a list of lists efficiently without unnecessary copying.
*   **String Methods:** Use `startswith()` and `endswith()` with a tuple of strings to check for multiple prefixes or suffixes at once.
*   **Decorators:** Use decorators to add common functionality (like logging, timing, caching) to functions without modifying their core logic. Use `functools.wraps()` to preserve the original function's metadata.
*   **Context Managers:** Use `with` statements and context managers (from `contextlib` or custom classes with `__enter__`/`__exit__`) to ensure resources are properly initialized and torn down, even in the presence of exceptions.
*   **Else Clauses:** Utilize the `else` clause in `try/except` blocks (runs if no exception), and in `for/while` loops (runs if the loop completes without a `break`) to write more expressive and less error-prone code.
*   **Single Assignment:** Prefer single-assignment form (assign to a variable once) over assign-and-mutate to reduce bugs and improve readability. Use conditional expressions where appropriate.
*   **Equality vs. Identity:** Use `is` or `is not` for singleton comparisons (e.g., `None`, `True`, `False`). Use `==` for value comparison.
*   **Object Comparisons:** When implementing custom classes, be careful with `__eq__`. Return `NotImplemented` for unhandled types. Consider edge cases like subclasses and hashing. Prefer using `attrs` or `dataclasses` to handle this automatically.
*   **Hashing:** If objects are equal, their hashes must be equal. Ensure attributes used in `__hash__` are immutable. Disable hashing with `__hash__ = None` if custom `__eq__` is implemented without a proper `__hash__`.
*   **`__init__()` vs. `__new__()`:** `__new__()` creates the object, `__init__()` initializes it. For immutable types, modifications must happen in `__new__()`.
*   **Default Arguments:** NEVER use mutable default arguments. Use `None` as a sentinel value instead.
*   **`__add__()` vs. `__iadd__()`:** `x += y` (in-place add) can modify the object in-place if `__iadd__` is implemented (like for lists), while `x = x + y` creates a new object. This matters when multiple variables reference the same object.
*   **Properties:** Use `@property` to create getters and setters only when needed, maintaining a simple attribute access syntax. Avoid properties for computationally expensive operations or those that can fail.
*   **Modules for Namespacing:** Use modules as the primary mechanism for grouping and namespacing code elements, not classes. Avoid `@staticmethod` and methods that don't use `self`.
*   **Argument Passing:** Python is call-by-value, where the values are object references (pointers). Assignment binds a name to an object. Modifying a mutable object through one name affects all names bound to it.
*   **Keyword/Positional Arguments:** Use `*` to force keyword-only arguments and `/` to force positional-only arguments. This can prevent argument transposition errors and make APIs clearer, especially for functions with multiple arguments of the same type.
*   **Type Hinting:** Annotate code with types to improve readability, debuggability, and maintainability. Use abstract types from `collections.abc` for container annotations (e.g., `Sequence`, `Mapping`, `Iterable`). Annotate return values, including `None`. Choose the most appropriate abstract type for function arguments and return types.
*   **`NewType`:** Use `typing.NewType` to create distinct types from primitives (like `int` or `str`) to prevent argument transposition and improve type safety.
*   **`__repr__()` vs. `__str__()`:** Implement `__repr__()` for unambiguous, developer-focused string representations, ideally evaluable. Implement `__str__()` for human-readable output. `__str__()` defaults to `__repr__()`.
*   **F-string Debug:** Use `f"{expr=}"` for concise debug printing, showing both the expression and its value.

### Libraries and Tools

*   **`collections.Counter`:** Use for efficiently counting hashable objects in an iterable.
*   **`collections.defaultdict`:** Useful for avoiding key checks when initializing dictionary values, e.g., appending to lists.
*   **`heapq`:** Use `heapq.nlargest()` and `heapq.nsmallest()` for efficiently finding the top/bottom N items. Use `heapq.merge()` to merge multiple sorted iterables.
*   **`attrs` / `dataclasses`:** Use these libraries to easily define simple classes with boilerplate methods like `__init__`, `__repr__`, `__eq__`, etc., automatically generated.
*   **NumPy:** Use NumPy for efficient array computing, element-wise operations, math functions, filtering, and aggregations on numerical data.
*   **Pandas:** When constructing DataFrames row by row, append to a list of dicts and call `pd.DataFrame()` once to avoid inefficient copying. Use `TypedDict` or `dataclasses` for intermediate row data.
*   **Flags:** Use libraries like `argparse` or `click` for command-line flag parsing. Access flag values in a type-safe manner.
*   **Serialization:** For cross-language serialization, consider JSON (built-in), Protocol Buffers, or msgpack. For Python serialization with validation, use `pydantic` for runtime validation and automatic (de)serialization, or `cattrs` for performance-focused (de)serialization with `dataclasses` or `attrs`.
*   **Regular Expressions:** Use `re.VERBOSE` to make complex regexes more readable with whitespace and comments. Choose the right method (`re.search`, `re.fullmatch`). Avoid regexes for simple string checks (`in`, `startswith`, `endswith`). Compile regexes used multiple times with `re.compile()`.
*   **Caching:** Use `functools.lru_cache` with care. Prefer immutable return types. Be cautious when memoizing methods, as it can lead to memory leaks if the instance is part of the cache key; consider `functools.cached_property`.
*   **Pickle:** Avoid using `pickle` due to security risks and compatibility issues. Prefer JSON, Protocol Buffers, or msgpack for serialization.
*   **Multiprocessing:** Be aware of potential issues with `multiprocessing` on some platforms, especially concerning `fork`. Consider alternatives like threads (`concurrent.futures.ThreadPoolExecutor`) or `asyncio` for I/O-bound tasks.
*   **Debugging:** Use `IPython.embed()` or `pdb.set_trace()` to drop into an interactive shell for debugging. Use visual debuggers if available. Log with context, including inputs and exception info using `logging.exception()` or `exc_info=True`.
*   **Property-Based Testing & Fuzzing:** Use `hypothesis` for property-based testing that generates test cases automatically. For coverage-guided fuzzing, consider `atheris` or `python-afl`.

---

## Testing Best Practices

### Testing Philosophy

**Use real code over mocks:** ADK tests should use real implementations as much as possible instead of mocking. Only mock external dependencies like network calls or cloud services.

**Test interface behavior, not implementation details:** Tests should verify that the public API behaves correctly, not how it's implemented internally. This makes tests resilient to refactoring and ensures the contract with users remains intact.

### Pytest Best Practices

*   **Assertions:** Use pytest's native `assert` statements with informative expressions. Pytest automatically provides detailed failure messages showing the values involved. Add custom messages with `assert condition, "helpful message"` when the expression alone isn't clear.
*   **Custom Assertions:** Write reusable helper functions (not methods) for repeated complex checks. Use `pytest.fail("message")` to explicitly fail a test with a custom message.
*   **Parameterized Tests:** Use `@pytest.mark.parametrize` to reduce duplication when running the same test logic with different inputs. This is more idiomatic than the `parameterized` library.
*   **Fixtures:** Use pytest fixtures (with `@pytest.fixture`) for test setup, teardown, and dependency injection. Fixtures are cleaner than class-based setup methods and can be easily shared across tests.
*   **Mocking:** Use `mock.create_autospec()` with `spec_set=True` to create mocks that match the original object's interface, preventing typos and API mismatch issues. Use context managers (`with mock.patch(...)`) to manage mock lifecycles and ensure patches are stopped. Prefer injecting dependencies via fixtures over patching.
*   **Asserting Mock Calls:** Use `mock.ANY` and other matchers for partial argument matching when asserting mock calls (e.g., `assert_called_once_with`).
*   **Temporary Files:** Use pytest's `tmp_path` and `tmp_path_factory` fixtures for creating isolated and automatically cleaned-up temporary files/directories. These are preferred over the `tempfile` module in pytest tests.
*   **Avoid Randomness:** Do not use random number generators to create inputs for unit tests. This leads to flaky, hard-to-debug tests. Instead, use deterministic, easy-to-reason-about inputs that cover specific behaviors.
*   **Test Invariants:** Focus tests on the invariant behaviors of public APIs, not implementation details.
*   **Test Organization:** Prefer simple test functions over class-based tests unless you need to share fixtures across multiple test methods in a class. Use descriptive test names that explain the behavior being tested.

### Unit Tests

**Quick start:** Run all tests with:
```bash
pytest tests/unittests
```

**Additional options:**
```bash
# Run tests in parallel for faster execution
pytest tests/unittests -n auto

# Run a specific test file during development
pytest tests/unittests/agents/test_llm_agent.py
```

---

## Error Handling

*   **Re-raising Exceptions:** Use a bare `raise` to re-raise the current exception, preserving the original stack trace. Use `raise NewException from original_exception` to chain exceptions, providing context. Use `raise NewException from None` to suppress the original exception's context.
*   **Exception Messages:** Always include a descriptive message when raising exceptions.
*   **Converting Exceptions to Strings:** `str(e)` can be uninformative. `repr(e)` is often better. For full details including tracebacks and chained exceptions, use functions from the `traceback` module (e.g., `traceback.format_exception(e)`, `traceback.format_exc()`).
*   **Terminating Programs:** Use `sys.exit()` for expected terminations. Uncaught non-`SystemExit` exceptions should signal bugs. Avoid functions that cause immediate, unclean exits like `os.abort()`.
*   **Returning None:** Be consistent. If a function can return a value, all paths should return a value (use `return None` explicitly). Bare `return` is only for early exit in conceptually void functions (annotated with `-> None`).

---

## Security Best Practices

1. **Input Validation**: Validate all tool inputs
2. **Least Privilege**: Tools should have minimal necessary permissions
3. **Secrets Management**: Use Secret Manager, never commit secrets
4. **Rate Limiting**: Implement rate limits on expensive operations
5. **Audit Logging**: Log all important operations
6. **Sanitization**: Sanitize outputs, especially when executing code
7. **Authentication**: Verify user identity and permissions

---

## Anti-Patterns to Avoid

### ❌ Hardcoding Domain Knowledge

```python
# Bad
instruction = "Analyze the users table with columns: id, name, email, created_at"
```

### ❌ Overly Specific Agents

```python
# Bad - too specific
customer_refund_analyzer = Agent(...)

# Good - general purpose
transaction_analyzer = Agent(...)
```

### ❌ Mixing Concerns

```python
# Bad - agent handles auth, execution, and formatting
def super_agent_tool():
    auth = authenticate()  # Should be separate
    result = execute()     # This is fine
    formatted = format()   # Should be separate or in prompt
```

### ❌ Ignoring Errors

```python
# Bad
def tool():
    result = risky_operation()
    return result

# Good
def tool():
    try:
        result = risky_operation()
        return result
    except SpecificError as e:
        return f"Error: {str(e)}. Please try X or Y."
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Tools Independently

```python
# Test tools without the agent
result = my_tool("test input")
print(result)
```

### Use ADK Web Interface

```bash
adk web --reload_agents
# Interactive testing with UI
```

### Check RAG Retrieval

```python
# Verify RAG is returning relevant context
context = query_rag_corpus("test question")
print(f"Retrieved {len(context)} characters")
print(context[:500])  # Preview
```

---

## Deployment

### To AgentSpace

```bash
# Deploy to AgentSpace
gcloud agent-space agents deploy \
  --project=$PROJECT_ID \
  --location=$LOCATION \
  --app-id=$APP_ID
```

### To Vertex AI Agent Engine

```bash
# Use project-specific deployment scripts
# See deploy/ directory for CLI tools
```

---

## Additional Resources

### Primary Resources

- **ADK Repository**: https://github.com/google/adk-python
- **ADK Documentation**: https://google.github.io/adk-docs
- **ADK Samples**: https://github.com/google/adk-samples
- **Vertex AI Docs**: https://cloud.google.com/vertex-ai/docs
- **AgentSpace Docs**: https://cloud.google.com/agentspace/docs
- **Gemini API Docs**: https://ai.google.dev/docs

### ADK Documentation

- [About ADK](https://github.com/google/adk-docs/blob/main/docs/get-started/about.md)
- [Installation](https://github.com/google/adk-docs/blob/main/docs/get-started/installation.md)
- [Quickstart](https://github.com/google/adk-docs/blob/main/docs/get-started/quickstart.md)
- [Agents Overview](https://github.com/google/adk-docs/blob/main/docs/agents/index.md)
- [Multi-Agent Systems](https://github.com/google/adk-docs/blob/main/docs/agents/multi-agents.md)
- [Tools Overview](https://github.com/google/adk-docs/blob/main/docs/tools/index.md)
- [Custom Tools](https://github.com/google/adk-docs/blob/main/docs/tools/custom-tools.md)
- [Deployment](https://github.com/google/adk-docs/blob/main/docs/deploy/index.md)
- [Streaming](https://github.com/google/adk-docs/blob/main/docs/streaming/index.md)
- [Sessions](https://github.com/google/adk-docs/blob/main/docs/sessions/index.md)
- [Observability](https://github.com/google/adk-docs/blob/main/docs/observability/index.md)
