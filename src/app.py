import streamlit as st
import os
import re
from loader import EmbeddingLoader
from chat import ChatEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_config_from_env():
    """Load model configuration from environment variables.
    
    Returns a tuple of (config_dict, is_configured) where is_configured is True
    if at least CHAT_PROVIDER is set via environment.
    """
    chat_provider = os.getenv("CHAT_PROVIDER")
    
    if not chat_provider:
        return None, False
    
    # Chat configuration
    chat_api_key = os.getenv("CHAT_API_KEY", "")
    chat_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    chat_azure_endpoint = os.getenv("CHAT_AZURE_ENDPOINT", "")
    chat_api_version = os.getenv("CHAT_API_VERSION", "2023-05-15")
    
    # Embedding configuration
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", chat_provider)
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embedding_azure_endpoint = os.getenv("EMBEDDING_AZURE_ENDPOINT", "")
    embedding_api_version = os.getenv("EMBEDDING_API_VERSION", "2023-05-15")
    
    # Build chat kwargs
    chat_kwargs = {}
    if chat_provider == "Azure OpenAI":
        chat_kwargs["azure_endpoint"] = chat_azure_endpoint
        chat_kwargs["api_version"] = chat_api_version
    
    # Build embedding kwargs
    embedding_kwargs = {}
    if embedding_provider == "Azure OpenAI":
        embedding_kwargs["azure_endpoint"] = embedding_azure_endpoint
        embedding_kwargs["api_version"] = embedding_api_version
    
    # Use chat API key for embedding if not explicitly set
    final_embedding_key = embedding_api_key if embedding_api_key else chat_api_key
    
    config = {
        "chat_provider": chat_provider,
        "chat_api_key": chat_api_key,
        "chat_model": chat_model,
        "chat_kwargs": chat_kwargs,
        "embedding_provider": embedding_provider,
        "embedding_api_key": final_embedding_key,
        "embedding_model": embedding_model,
        "embedding_kwargs": embedding_kwargs,
    }
    
    return config, True


def render_math(content: str) -> str:
    """Convert LaTeX math notation to Streamlit-compatible format.
    
    Converts:
    - \\[...\\] -> $$...$$ (block math)
    - \\(...\\) -> $...$ (inline math)
    """
    # Convert block math: \[...\] -> $$...$$
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    # Convert inline math: \(...\) -> $...$
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content, flags=re.DOTALL)
    return content

st.set_page_config(page_title="Repo Chatbot", layout="wide")

@st.cache_resource
def load_engine(repo_path):
    loader = EmbeddingLoader(repo_path)
    chunks = loader.load()
    if not chunks:
        return None
    return ChatEngine(chunks)

st.title("Repo Chatbot")

repo_path = os.getenv("REPO_PATH", "/data")
st.write(f"Reading from: {repo_path}")

# Check if model configuration is provided via environment variables
env_config, models_preconfigured = get_model_config_from_env()

if models_preconfigured:
    # Use configuration from environment variables
    chat_provider = env_config["chat_provider"]
    chat_api_key = env_config["chat_api_key"]
    chat_model = env_config["chat_model"]
    chat_kwargs = env_config["chat_kwargs"]
    embedding_provider = env_config["embedding_provider"]
    final_embedding_key = env_config["embedding_api_key"]
    embedding_model_name = env_config["embedding_model"]
    embedding_kwargs = env_config["embedding_kwargs"]
    
    # Show a notice in the sidebar that models are pre-configured
    st.sidebar.info("Model configuration is set via environment variables.")
    st.sidebar.write(f"**Chat Provider:** {chat_provider}")
    st.sidebar.write(f"**Chat Model:** {chat_model}")
    st.sidebar.write(f"**Embedding Provider:** {embedding_provider}")
    st.sidebar.write(f"**Embedding Model:** {embedding_model_name}")
else:
    # Sidebar Config - show full configuration UI
    st.sidebar.header("Chat Configuration")
    chat_provider = st.sidebar.selectbox("Chat Provider", ["OpenAI", "Azure OpenAI", "Anthropic", "Mock"])
    chat_api_key = st.sidebar.text_input("Chat API Key", type="password")

    chat_kwargs = {}
    default_llm = "gpt-3.5-turbo"

    if chat_provider == "Azure OpenAI":
        st.sidebar.subheader("Azure Chat Settings")
        chat_azure_endpoint = st.sidebar.text_input("Chat Azure Endpoint", help="https://your-resource.openai.azure.com/")
        chat_api_version = st.sidebar.text_input("Chat API Version", value="2023-05-15")
        # For Azure, the model name is often the deployment name
        default_llm = "my-gpt-deployment"
        chat_kwargs["azure_endpoint"] = chat_azure_endpoint
        chat_kwargs["api_version"] = chat_api_version
    elif chat_provider == "Anthropic":
        default_llm = "claude-3-opus-20240229"

    chat_model = st.sidebar.text_input("Chat Model / Deployment Name", value=default_llm)

    st.sidebar.divider()

    st.sidebar.header("Embedding Configuration")
    st.sidebar.info("Must match the provider/model used to generate the index.")
    embedding_provider = st.sidebar.selectbox("Embedding Provider", ["OpenAI", "Azure OpenAI", "Mock"])
    embedding_api_key = st.sidebar.text_input("Embedding API Key", type="password", help="Leave empty if same as Chat API Key (if provider matches).")

    embedding_kwargs = {}
    embedding_default_model = "text-embedding-ada-002"

    if embedding_provider == "Azure OpenAI":
        st.sidebar.subheader("Azure Embedding Settings")
        # If users reuse Chat Azure settings, they can just copy/paste or we could add a checkbox "Same as Chat".
        # For flexibility, let's keep separate but maybe default to empty and logic handle it? No, explicit is better.
        emb_azure_endpoint = st.sidebar.text_input("Embedding Azure Endpoint", help="https://your-resource.openai.azure.com/")
        emb_api_version = st.sidebar.text_input("Embedding API Version", value="2023-05-15")
        embedding_default_model = "my-embedding-deployment"
        embedding_kwargs["azure_endpoint"] = emb_azure_endpoint
        embedding_kwargs["api_version"] = emb_api_version

    embedding_model_name = st.sidebar.text_input("Embedding Model / Deployment Name", value=embedding_default_model)

    # Resolve Keys
    final_embedding_key = embedding_api_key if embedding_api_key else chat_api_key

# Load Index
with st.spinner("Loading index..."):
    engine = load_engine(repo_path)

if not engine:
    st.error(f"No embeddings found in {repo_path}/.copilot-index. Please generate embeddings first.")
    st.stop()

st.sidebar.success(f"Loaded {len(engine.chunks)} chunks.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(render_math(message["content"]))

if prompt := st.chat_input("Ask a question about the repo"):
    # Validation
    if chat_provider != "Mock" and not chat_api_key:
        st.error("Please provide a Chat API Key.")
    elif embedding_provider != "Mock" and not final_embedding_key:
        st.error("Please provide an Embedding API Key (or use Chat API Key if same provider).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(render_math(prompt))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Get embedding
                    query_embedding = engine.get_embedding(
                        prompt,
                        embedding_provider,
                        final_embedding_key,
                        embedding_model=embedding_model_name,
                        **embedding_kwargs
                    )

                    # 2. Search
                    results = engine.search(query_embedding)

                    # 3. Generate
                    response = engine.generate_response(
                        prompt,
                        results,
                        chat_provider,
                        chat_api_key,
                        model=chat_model,
                        **chat_kwargs
                    )

                    st.markdown(render_math(response))

                    with st.expander("Retrieved Context"):
                        for r in results:
                            st.markdown(f"**Source:** {r.source}")
                            st.markdown(render_math(r.content[:500] + ("..." if len(r.content) > 500 else "")))
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Chat error")
