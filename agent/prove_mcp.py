"""Exercise the real stdio server with the SDK client, without network APIs."""
import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.run_rules import write
from agent.tools import ROOT


async def main():
    params = StdioServerParameters(command=sys.executable, args=['-I',str(ROOT/'agent/mcp_server.py')], cwd=str(ROOT))
    with (ROOT/'data/claims/mcp_server.log').open('w') as errors:
        async with stdio_client(params,errlog=errors) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                await session.initialize()
                listed=await session.list_tools()
                calls=[]
                for name,args in [('ceiling_heights',{'sid':'VN4b1MGnnCR'}),('search_homes',{'filters':{'city':'Houston','min_ceiling_height_m':3.35,'limit':2}})]:
                    result=await session.call_tool(name,args)
                    assert not result.isError
                    calls.append(dict(tool=name,arguments=args,result=result.model_dump(mode='json')))
                write(ROOT/'data/claims/mcp_proof.json',dict(transport='stdio',tool_names=[t.name for t in listed.tools],calls=calls,passed=True))
                print(json.dumps(dict(passed=True,tools=len(listed.tools),calls=len(calls))))


if __name__=='__main__':asyncio.run(main())
