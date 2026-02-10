"""
MCP Server 创建与 tool 注册
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from app.services.mcp.search import GrokSearchService


def create_mcp_server() -> FastMCP:
    """创建 MCP 服务器并注册搜索工具"""
    mcp = FastMCP(
        name="grok-web-search",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )

    @mcp.tool()
    async def grok_web_search(query: str, model: str = "") -> CallToolResult:
        """Search the web using Grok's native search capability.

        Args:
            query: The search query to execute.
            model: Optional model override (defaults to config value).
        """
        search_service = GrokSearchService()
        result = await search_service.search(query, model=model or None)
        return CallToolResult(content=[TextContent(type="text", text=result)])

    return mcp


__all__ = ["create_mcp_server"]
