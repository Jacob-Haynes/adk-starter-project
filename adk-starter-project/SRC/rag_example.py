import os
import uuid
from typing import List, Optional, Dict, Any

from vertexai import init as vertexai_init
from vertexai import rag
from dotenv import load_dotenv

load_dotenv()

# example rag set up script for Vertex AI RAG Engine - have a read of the functions as there are some useful comments
# particularly on managing corpus and file versions.
# You will need to initialise the engine in the root agent script to use it in any agent/sub-agent/or tool.

# config
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_REGION = os.getenv("GOOGLE_CLOUD_LOCATION")
FORCE_REIMPORT_CORPUS = (
    os.getenv("RAG_FORCE_REIMPORT", "default_string_value").lower() == "true"
)

DRIVE_RESOURCE_IDS = [
    "google-drive URL goes here, can be files or a folder"
]
# For Drive permissions, the user still needs to share Drive files with:
# service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com


# RAG settings
FIXED_RAG_CORPUS_DISPLAY_NAME = os.getenv("FIXED_RAG_CORPUS_DISPLAY_NAME") #if you want to use a fixed name for the corpus
DEFAULT_CORPUS_PREFIX = "example-rag-corpus" # Default prefix for generated corpus names
EMBEDDING_MODEL_ID = "text-embedding-005"
EMBEDDING_MODEL_PUBLISHER = "google"


# Vertex AI RAG Engine
class VertexRAGEngine:
    """
    Manages a Vertex AI RAG Corpus
    """

    def __init__(self, project_id: str, region: str):
        if not project_id or not region:
            raise ValueError("ERROR: Project ID or Region must be set")
        self.project_id = project_id
        self.region = region

        # Initialise Vertex AI SDK
        try:
            vertexai_init(project=self.project_id, location=self.region)
            print(
                f"INFO: Vertex AI SDK initialised for project {project_id} in {region}."
            )
        except Exception as e:
            print(
                f"ERROR: Error initializing Vertex AI SDK: {e}. Confirm you are authenticated"
            )
            raise

        self.corpus_name: Optional[str] = None
        self.corpus_display_name: Optional[str] = None

    def _generate_unique_corpus_display_name(self) -> str:
        return f"{DEFAULT_CORPUS_PREFIX}-{str(uuid.uuid4())[:8]}"

    def get_or_create_corpus(
        self, display_name_override: Optional[str] = None
    ) -> tuple[Optional[str], bool]:
        """
        Gets an existing RAG corpus by display name or creates a new one.
        Prioritises display_name_override, then FIXED_RAG_CORPUS_DISPLAY_NAME from .env.
        If neither is specific, it generates a unique name for creation.
        Returns a tuple: (corpus_resource_name, was_newly_created_bool).
        """
        env_fixed_name = FIXED_RAG_CORPUS_DISPLAY_NAME
        chosen_display_name: Optional[str] = None
        attempt_to_find_specific_named_corpus = False
        was_newly_created = False

        if display_name_override:
            chosen_display_name = display_name_override
            attempt_to_find_specific_named_corpus = True
            print(
                f"INFO: Using display_name_override for corpus: '{chosen_display_name}'"
            )
        elif env_fixed_name:
            chosen_display_name = env_fixed_name
            attempt_to_find_specific_named_corpus = True
            print(
                f"INFO: Attempting to use FIXED_RAG_CORPUS_DISPLAY_NAME from .env: '{chosen_display_name}'"
            )
        else:
            chosen_display_name = self._generate_unique_corpus_display_name()
            print(
                f"INFO: No override or .env fixed name. Using generated name for corpus: '{chosen_display_name}'"
            )

        if not chosen_display_name:
            print("ERROR: Could not determine a display name for the corpus.")
            return None, False

        self.corpus_display_name = chosen_display_name

        if attempt_to_find_specific_named_corpus:
            print(
                f"INFO: Checking if corpus with display name '{chosen_display_name}' already exists..."
            )
            try:
                all_listed_corpora = list(rag.list_corpora())
                print(
                    f"INFO: rag.list_corpora() (fully paged) found {len(all_listed_corpora)} corpora in region '{self.region}'."
                )

                found_corpora_matching_display_name = []
                for i, corp_resource in enumerate(all_listed_corpora):
                    print(
                        f"INFO: Checking listed corpus #{i}: Name='{corp_resource.name}', DisplayName='{corp_resource.display_name}' against TargetDisplayName='{chosen_display_name}'"
                    )
                    if corp_resource.display_name == chosen_display_name:
                        found_corpora_matching_display_name.append(corp_resource)

                if len(found_corpora_matching_display_name) == 1:
                    self.corpus_name = found_corpora_matching_display_name[0].name
                    print(
                        f"SUCCESS: Found exactly one existing RAG Corpus: '{chosen_display_name}' with resource name: {self.corpus_name}"
                    )
                    return self.corpus_name, False
                elif len(found_corpora_matching_display_name) > 1:
                    self.corpus_name = found_corpora_matching_display_name[0].name
                    print(
                        f"WARNING: Found {len(found_corpora_matching_display_name)} corpora with the display name '{chosen_display_name}'. This is ambiguous."
                    )
                    print(
                        f"INFO: Using the first one found: {self.corpus_name}. Consider cleaning up duplicates in the Cloud Console or using unique display names."
                    )
                    return self.corpus_name, False
                else:
                    print(
                        f"INFO: Corpus with display name '{chosen_display_name}' not found among existing corpora. Will proceed to create it."
                    )
            except Exception as list_e:
                print(
                    f"ERROR: Error listing corpora when checking for '{chosen_display_name}': {list_e}. Proceeding to attempt creation."
                )

            print(
                f"INFO: Attempting to create RAG Corpus with display name: '{chosen_display_name}'"
            )
            try:
                created_corpus_resource = rag.create_corpus(
                    display_name=chosen_display_name
                )
                self.corpus_name = created_corpus_resource.name
                was_newly_created = True
                print(
                    f"SUCCESS: Created RAG Corpus: '{chosen_display_name}' with resource name: {self.corpus_name}"
                )
                return self.corpus_name, was_newly_created
            except Exception as e:
                print(
                    f"ERROR: Failed during rag.create_corpus for '{chosen_display_name}': {e}"
                )
                if attempt_to_find_specific_named_corpus:
                    print(
                        f"INFO: Re-checking for '{chosen_display_name}' after explicit creation attempt failed..."
                    )
                    try:
                        all_listed_corpora_after_fail = list(rag.list_corpora())
                        for corp_resource in all_listed_corpora_after_fail:
                            if corp_resource.display_name == chosen_display_name:
                                self.corpus_name = corp_resource.name
                                self.corpus_display_name = corp_resource.display_name
                                print(
                                    f"SUCCESS: Found existing RAG Corpus '{chosen_display_name}' after failed create attempt: {self.corpus_name}"
                                )
                                return self.corpus_name, False
                        print(
                            f"INFO: Corpus '{chosen_display_name}' still not found after re-checking."
                        )
                    except Exception as list_e_after_fail:
                        print(f"ERROR: Error re-listing corpora: {list_e_after_fail}")
        return None, False

    def import_document(self, corpus_resource_name: str, source_paths: List[str]):
        """Imports files from Google Drive URIs into the RAG corpus."""
        if not source_paths:
            print("INFO: No source files provided.")
            return
        print(
            f"INFO: Importing {len(source_paths)} files from {source_paths} into corpus {corpus_resource_name}."
        )

        try:
            transformation_cfg = rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=512,
                    chunk_overlap=100,
                )
            )
            response = rag.import_files(
                corpus_name=corpus_resource_name,
                paths=source_paths,
                transformation_config=transformation_cfg,
            )
            print(f"Import files response: {response}")
            print(
                f"INFO: Data import process initiated for corpus '{corpus_resource_name}'."
                "This may take some time depending on the volume of data."
            )

        except Exception as e:
            print(f"ERROR: Error importing files: {e}")
            print(
                "Common issues: Ensure Drive files/folders are shared with the RAG service agent, "
                "URIs are valid, and the RAG API is enabled with necessary permissions."
            )
            raise

    def setup_rag_corpus_and_import_files(
        self,
        source_items: List[str],
        target_corpus_display_name: Optional[str] = None,
        force_reimport_if_corpus_exists: bool = False,
    ) -> bool:
        """
        Ensures a corpus exists and specified Drive/GCS documents are imported.
        Imports documents only if the corpus was newly created by this call,
        or if force_reimport_if_corpus_exists is True.
        """
        if not source_items:
            print("INFO: No source items provided.")
            return False
        corpus_resource_name, was_newly_created = self.get_or_create_corpus(
            display_name_override=target_corpus_display_name
        )
        if not corpus_resource_name:
            print("INFO: Failed to get or create RAG Corpus.")
            return False
        # Decide if to import documents
        should_import_documents = False
        if was_newly_created:
            should_import_documents = True
            print(
                f"INFO: Corpus '{corpus_resource_name}' was newly created. Proceeding to import documents."
            )
        elif force_reimport_if_corpus_exists:
            should_import_documents = True
            print(
                f"INFO: Corpus '{corpus_resource_name}' already exists, but 'force_reimport_if_corpus_exists' is True. Proceeding with document import."
            )
        else:
            print(
                f"INFO: Corpus '{corpus_resource_name}' already exists and 'force_reimport_if_corpus_exists' is False. Skipping document import."
            )

        if should_import_documents:
            try:
                self.import_document(corpus_resource_name, source_items)
                print(
                    f"INFO: Document import process initiated for corpus '{corpus_resource_name}'."
                )
            except Exception as e:
                print(f"ERROR: Document import failed: {e}")
                return False
        return True

    def query_corpus(
        self,
        text_query: str,
        top_k: int = 3,
        vector_distance_threshold: Optional[float] = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Queries the RAG corpus using 'rag.retrieval_query'.
        :param text_query:
        :param top_k: defaults to 3
        :param vector_distance_threshold: defaults to 0.5
        :return: list of context dictionaries
        """
        if not self.corpus_name:
            print(
                "ERROR: Corpus resource name not set. Cannot Query. Ensure RAG setup was successful"
            )
            if not self.get_or_create_corpus(
                display_name_override=FIXED_RAG_CORPUS_DISPLAY_NAME
            ):
                print("ERROR: Failed to get or create RAG Corpus.")
                return []

        print(f"INFO: Querying RAG Corpus {self.corpus_name} for: '{text_query}'.")

        retrieval_cfg = rag.RagRetrievalConfig(
            top_k=top_k,
            filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),
        )
        try:
            retrieve_contexts_response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=self.corpus_name,
                        # rag_file_ids = [...] # Optional: to query specific files within the corpus
                    )
                ],
                text=text_query,
                rag_retrieval_config=retrieval_cfg,
            )

            processed_results = []
            if not retrieve_contexts_response:
                print("DEBUG: rag.retrieval_query returned a None or empty response.")
                return processed_results

            if (
                not hasattr(retrieve_contexts_response, "contexts")
                or not retrieve_contexts_response.contexts
            ):
                print(
                    "DEBUG: The RetrieveContextsResponse object has no 'contexts' attribute or it's empty."
                )
                return processed_results

            rag_contexts_container = retrieve_contexts_response.contexts
            # print(f"DEBUG: Type of rag_contexts_container (from response.contexts): {type(rag_contexts_container)}") # Should be RagContexts

            if not rag_contexts_container:
                print("DEBUG: The rag_contexts_container (response.contexts) is None.")
                return processed_results

            if (
                not hasattr(rag_contexts_container, "contexts")
                or not rag_contexts_container.contexts
            ):
                print(
                    "DEBUG: The RagContexts object (rag_contexts_container) has no inner 'contexts' list or it's empty."
                )
                return processed_results

            individual_context_list = rag_contexts_container.contexts
            # print(f"DEBUG: Type of individual_context_list (from rag_contexts_container.contexts): {type(individual_context_list)}")

            if not individual_context_list:
                print("DEBUG: The final individual_context_list is None or empty.")
                return processed_results

            print(
                f"SUCCESS: Successfully retrieved {len(individual_context_list)} raw context items. Processing them..."
            )
            for i, ctx_item in enumerate(individual_context_list):
                # print(f"DEBUG: Processing item #{i}: type {type(ctx_item)}")

                text = getattr(ctx_item, "text", "")
                source_uri = getattr(ctx_item, "source_uri", "")
                source_display_name = getattr(ctx_item, "source_display_name", "")
                distance_or_score = getattr(
                    ctx_item, "distance", getattr(ctx_item, "score", 0.0)
                )  # Check for distance, fallback to score

                processed_results.append(
                    {
                        "text": text,
                        "source_uri_display": source_display_name
                        or source_uri,  # Prefer display name, fallback to URI
                        "distance": distance_or_score,
                    }
                )

            return processed_results
        except Exception as e:
            print(f"Error querying RAG Corpus: {e}")
            return []


# Main block for standalone RAG setup
if __name__ == "__main__":
    print("--- RAG  Engine (vertexai.rag SDK) Standalone Setup/Test ---")
    if not GCP_PROJECT_ID:
        print("ERROR: No Google Cloud Project ID provided.")
    elif not DRIVE_RESOURCE_IDS:
        print("ERROR: No Drive Resource IDs provided.")
    else:
        print(
            f"Initialising RAG Engine for project '{GCP_PROJECT_ID}' in region: {GCP_REGION}"
        )
        print(
            f"Agent RAG initialisation using RAG_FORCE_REIMPORT value: {FORCE_REIMPORT_CORPUS}"
        )
        engine = VertexRAGEngine(project_id=GCP_PROJECT_ID, region=GCP_REGION)
        setup_successful = engine.setup_rag_corpus_and_import_files(
            source_items=DRIVE_RESOURCE_IDS,
            target_corpus_display_name=FIXED_RAG_CORPUS_DISPLAY_NAME,
            force_reimport_if_corpus_exists=FORCE_REIMPORT_CORPUS,
        )

        if setup_successful and engine.corpus_name:
            print("\n--- RAG Engine Setup Complete! ---")
            print(
                f"Using RAG Corpus: {engine.corpus_name} (Display: {engine.corpus_display_name})"
            )
            sample_query = "How can I change my profile information?"
            print(f"\nTesting RAG engine with sample query: '{sample_query}'")
            retrieved_contexts = engine.query_corpus(sample_query)
            if retrieved_contexts:
                print(f"\nRetrieved {len(retrieved_contexts)} contexts:")
                for i, context_dict in enumerate(retrieved_contexts):
                    print(f"\n--- Context {i+1} ---")
                    print(
                        f"Source URI Display: {context_dict.get('source_uri_display', 'N/A')}"
                    )
                    print(f"Content Text: {context_dict.get('text', 'N/A')}")
                    print(f"Distance: {context_dict.get('distance', 0.0):.4f}")
            else:
                print("No retrieved contexts found.")
        else:
            print("\nRAG Engine setup failed or corpus is not ready")
