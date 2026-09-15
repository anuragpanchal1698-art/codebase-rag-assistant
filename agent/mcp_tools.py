"""
Real MCP integration: connects to the official GitHub MCP server
(@modelcontextprotocol/server-github) via stdio, exposing GitHub operations
(issues, PRs, repo search, file contents) as genuine MCP-protocol tools.

Tool outputs are wrapped to truncate large responses - GitHub search results
can easily return tens of thousands of tokens of raw JSON, which blows past
Groq's free-tier 8000 TPM limit. Truncating keeps enough info to be useful
while staying within budget.
"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from config import GITHUB_TOKEN

_mcp_client = None
_mcp_tools_cache = None

MAX_OUTPUT_CHARS = 1500  # roughly ~400 tokens, keeps total request well under limit


def _wrap_tool_with_truncation(tool):
    """Wraps an MCP tool's coroutine so its string output gets truncated.
    Handles both plain string returns and (content, artifact) tuple returns
    (the latter used by tools with response_format='content_and_artifact')."""
    original_coroutine = tool.coroutine

    async def truncated_coroutine(*args, **kwargs):
        result = await original_coroutine(*args, **kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            content, artifact = result
            content_text = str(content)
            if len(content_text) > MAX_OUTPUT_CHARS:
                content_text = content_text[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
            return (content_text, artifact)

        text = str(result)
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n... [truncated - ask a more specific question for more detail]"
        return text

    tool.coroutine = truncated_coroutine
    return tool


async def get_mcp_tools():
    """Lazily connects to the GitHub MCP server and returns its tools
    as LangChain-compatible tool objects, with output truncation applied."""
    global _mcp_client, _mcp_tools_cache

    if _mcp_tools_cache is not None:
        return _mcp_tools_cache

    _mcp_client = MultiServerMCPClient(
        {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN or "",
                },
                "transport": "stdio",
            }
        }
    )

    raw_tools = await _mcp_client.get_tools()
    _mcp_tools_cache = [_wrap_tool_with_truncation(t) for t in raw_tools]
    return _mcp_tools_cache