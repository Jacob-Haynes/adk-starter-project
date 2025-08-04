"""Deployment script for the Vertex AI Agent Engine."""

import os
import vertexai

from vertexai import agent_engines
from absl import app, flags
from vertexai.preview.reasoning_engines import AdkApp
from dotenv import load_dotenv

from SRC.agent import root_agent

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")

flags.DEFINE_bool("list", False, "List all agents.")
flags.DEFINE_bool("create", False, "Creates a new agent.")
flags.DEFINE_bool("delete", False, "Deletes an existing agent.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete"])


def create() -> None:
    """Creates an agent engine."""
    adk_app = AdkApp(agent=root_agent, enable_tracing=True)

    extra_packages = ["./SRC/tools","./SRC/sub_agents, etc."]

    remote_agent = agent_engines.create(
        adk_app,
        display_name=root_agent.name,
        description=root_agent.description,
        requirements=[
            "your requirements go here"
        ],
        extra_packages=extra_packages,
        env_vars={
            "extra environment variables go here":"value",
            "do not use GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION as these are set by the SDK":"value",
        },
    )
    print(f"Created agent engine: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    """Deletes an agent engine."""
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def list_agents() -> None:
    """Lists all agent engines."""
    remote_agents = agent_engines.list()
    template = """
    {agent.name} ("{agent.display_name}")
    - Create time: {agent.create_time}
    - Update time: {agent.update_time}
    """
    remote_agents_string = "\n".join(
        template.format(agent=agent) for agent in remote_agents
    )
    print(f"All remote agents:\n{remote_agents_string}")


def main(argv: list[str]) -> None:
    """Main function to run the deployment script."""
    del argv
    load_dotenv()

    project_id = (
        FLAGS.project_id if FLAGS.project_id else os.getenv("GOOGLE_CLOUD_PROJECT")
    )
    location = FLAGS.location if FLAGS.location else os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = FLAGS.bucket if FLAGS.bucket else os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"BUCKET: {bucket}")

    if not project_id:
        print("Missing required environment variable: GOOGLE_CLOUD_PROJECT")
        return
    elif not location:
        print("Missing required environment variable: GOOGLE_CLOUD_LOCATION")
        return
    elif not bucket:
        print("Missing required environment variable: GOOGLE_CLOUD_STORAGE_BUCKET")
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if FLAGS.list:
        list_agents()
    elif FLAGS.create:
        create()
    elif FLAGS.delete:
        if not FLAGS.resource_id:
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    else:
        print("Unknown command")


if __name__ == "__main__":
    app.run(main)
