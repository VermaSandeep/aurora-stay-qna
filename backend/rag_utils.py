
from langchain_openai import ChatOpenAI
import os
import json
from pathlib import Path


def _load_openai_settings() -> tuple[str, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    config_path = Path(
        os.getenv(
            "RAG_CONFIG_PATH",
            str(Path(__file__).resolve().parents[1] / "config.json"),
        )
    )

    if (not api_key or not api_base) and config_path.is_file():
        try:
            with config_path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in RAG configuration file: {config_path}"
            ) from exc

        if not isinstance(config, dict):
            raise RuntimeError(
                f"RAG configuration file must contain a JSON object: {config_path}"
            )

        api_key = api_key or config.get("OPENAI_API_KEY")
        api_base = api_base or config.get("OPENAI_API_BASE")

    if not isinstance(api_key, str) or not api_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in the server environment "
            f"or provide it in {config_path}."
        )
    if api_base is not None and not isinstance(api_base, str):
        raise RuntimeError("OPENAI_API_BASE must be a string when provided.")

    return api_key, api_base

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant answering questions about medical insurance policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exclusions, limitations, exceptions, or coverage details.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent benefits, exclusions, limits, waiting periods, conditions, or coverage details.
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    api_key, api_base = _load_openai_settings()

    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
