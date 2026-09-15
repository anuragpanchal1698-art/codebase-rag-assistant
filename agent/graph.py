"""
Agentic RAG graph built with LangGraph.

Flow:
    START -> agent (LLM w/ tools bound) -> [tool calls?] -> tools -> agent (loop)
                                          -> [no tool calls] -> END

The LLM itself acts as the planner: given the system prompt below, it decides
whether to call search_codebase (RAG), local file retrieval, GitHub MCP tools,
or answer directly / say "I don't know".
This is what makes it "agentic" rather than a fixed retrieve-then-generate chain.

Tool loading is async because the GitHub MCP server is spawned as a
subprocess and its tools are fetched asynchronously via langchain-mcp-adapters.
"""
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL
from agent.tools import ALL_TOOLS as LOCAL_TOOLS
from agent.mcp_tools import get_mcp_tools
SYSTEM_PROMPT = """You are an assistant with tools for: searching the LangChain \
codebase, searching user-uploaded PDFs, GitHub data via MCP, and reading full files.

Tool selection:
- If the user's question could relate to a PDF they uploaded (mentions "pdf", "document", \
"story", "uploaded", "this", or asks something not obviously about code) - ALWAYS call \
search_pdf_documents FIRST to check, even if unsure. Never assume no PDF exists without checking.
- Code/architecture/docs questions about LangChain -> search_codebase
- GitHub issues/PRs/commits -> GitHub MCP tools (always scope with repo:langchain-ai/langchain)
- Need full file context -> get_file_content

Rules:
- Always cite the file path or PDF name (+ page if available).
- If search_pdf_documents returns "No uploaded PDFs found", only then say you couldn't \
find a PDF - don't assume this without calling the tool first.
- May call multiple tools across turns before answering.

Style: Write like a person explaining something in conversation, not a report. \
Short paragraphs (2-3 sentences), blank line between them. No JSON, no tables unless \
asked, minimal bold. Cite sources inline in a sentence, never as a page-number list \
at the end. Open with a direct one-sentence answer, then explain.
"""

async def build_agent():
    mcp_tools = await get_mcp_tools()
    all_tools = LOCAL_TOOLS + mcp_tools

    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(all_tools).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(all_tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


_compiled_agent = None
_agent_lock = asyncio.Lock()


async def get_agent():
    """Lazily builds and caches the compiled agent (async, since tool loading is async)."""
    global _compiled_agent
    if _compiled_agent is None:
        async with _agent_lock:
            if _compiled_agent is None:  # re-check after acquiring lock
                _compiled_agent = await build_agent()
    return _compiled_agent


def ask(question: str, chat_history: list = None):
    """Blocking wrapper around _ask_async - used by the CLI (main.py)."""
    return asyncio.run(_ask_async(question, chat_history))


async def _ask_async(question: str, chat_history: list = None):
    """Run a single question through the agent, optionally with prior history.
    Used by the CLI (via ask()) and the non-streaming API route directly."""
    agent = await get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]
    result = await agent.ainvoke({"messages": messages})
    return result["messages"][-1].content, result["messages"]


async def astream(question: str, chat_history: list = None):
    """Async generator yielding text chunks as the agent generates them.
    Used by the streaming API endpoint (/chat/stream)."""
    agent = await get_agent()
    messages = (chat_history or []) + [HumanMessage(content=question)]

    try:
        async for event in agent.astream_events(
            {"messages": messages}, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield chunk.content
    except Exception as e:
        yield f"\n\n[Streaming error: {str(e)}]"