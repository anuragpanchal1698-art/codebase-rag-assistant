"""
Test cases for evaluating retrieval accuracy.

Each case has a query and the expected file path substring that should
appear somewhere in the top-K retrieved results. Paths are relative to
the repo root (matching the `path` metadata field stored in Qdrant/BM25).

To add more cases: pick a real class/function you can verify in the repo,
write a natural question about it, and note its actual file location.
"""

TEST_CASES = [
    {
        "query": "Where is RecursiveCharacterTextSplitter defined?",
        "expected_path_contains": "character.py",
    },
    {
        "query": "What file contains the BaseRetriever class?",
        "expected_path_contains": "retrievers",
    },
    {
        "query": "Where is ConversationBufferMemory implemented?",
        "expected_path_contains": "memory",
    },
    {
        "query": "What file defines the BaseOutputParser class?",
        "expected_path_contains": "output_parsers",
    },
    {
        "query": "Where is the Document class defined?",
        "expected_path_contains": "documents",
    },
    {
        "query": "What file contains BaseChatModel?",
        "expected_path_contains": "chat_models",
    },
    {
        "query": "Where is the PromptTemplate class defined?",
        "expected_path_contains": "prompts",
    },
    {
        "query": "What file implements BaseTool?",
        "expected_path_contains": "tools",
    },
]