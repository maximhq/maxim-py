import os
import dotenv

from openai import OpenAI
from agents import (
    Agent,
    function_tool,
    WebSearchTool,
    FileSearchTool,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

dotenv.load_dotenv()

# --- Agent: Search Agent ---
search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def upload_file(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id, file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}


def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed,
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}


# --- Agent: Knowledge Agent ---
# Will be initialized with actual vector_store_id after creation
knowledge_agent = None


def create_knowledge_agent(vector_store_id: str) -> Agent:
    """Create the knowledge agent with the actual vector store ID."""
    return Agent(
        name="KnowledgeAgent",
        instructions=(
            "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
        ),
        tools=[
            FileSearchTool(
                max_num_results=3,
                vector_store_ids=[vector_store_id],
            ),
        ],
    )


# --- Tool 1: Fetch account information (dummy) ---
@function_tool
def get_account_info(user_id: str) -> dict:
    """Return dummy account info for a given user."""
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive",
    }


# --- Agent: Account Agent ---
account_agent = Agent(
    name="AccountAgent",
    instructions=(
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],
)

# --- Agent: Triage Agent ---
# Will be initialized after knowledge_agent is created
triage_agent = None


def create_triage_agent(knowledge_agent_instance: Agent) -> Agent:
    """Create the triage agent with all handoff agents."""
    return Agent(
        name="Assistant",
        instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Acme Shop. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- KnowledgeAgent for product FAQs
- SearchAgent for anything requiring real-time web search
"""),
        handoffs=[account_agent, knowledge_agent_instance, search_agent],
    )
