import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.genai import types

load_dotenv()

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_REGION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCP_MODEL = os.getenv("GCP_MODEL")

# --- Agent Definition ---

example_agent = LlmAgent(
    name="example-agent",
    description="This is a breif description other agents can use to understand what this agent does.",
    model=GCP_MODEL,
    instruction="This is where your prompt goes",
    tools=[],  # OPTIONAL List of tools this agent can use, e.g., [query_knowledge_base], can be any function or an agent as a tool
    # output_schema= OutputSchema, # OPTIONAL Pydantic model defining the output structure, if needed - mutually exclusive with tools
    output_key="output", # OPTIONAL The key in the session state where the agent's output will be stored
    generate_content_config=types.GenerateContentConfig(temperature=0.3, top_p=0.9, max_output_tokens=512), # OPTIONAL configuration for the generation process
)

# --- Agent Usage ---
root_agent = example_agent
# adk web - ran in project root will launch the agent in the browser UI