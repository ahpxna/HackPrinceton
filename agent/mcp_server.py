"""Read-only local claims tools over MCP stdio; stdout belongs to the protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server.fastmcp import FastMCP
from agent.tools import TOOL_FUNCTIONS
from agent.buyer_tools import BUYER_TOOLS

mcp = FastMCP('HackPrinceton local listing evidence')
for function in TOOL_FUNCTIONS+BUYER_TOOLS:
    mcp.tool()(function)

if __name__ == '__main__':
    mcp.run(transport='stdio')
