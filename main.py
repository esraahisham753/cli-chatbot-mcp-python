import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.huggingface import HuggingFace

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Hugging Face Config
hf_access_token = os.getenv("HF_ACCESS_TOKEN", "")
hf_model = os.getenv("HF_MODEL", "")


assert hf_model, "Error: HF_MODEL cannot be empty. Update .env"
assert hf_access_token, (
    "Error: HF_ACCESS_TOKEN cannot be empty. Update .env"
)


async def main():
    hf_service = HuggingFace(model=hf_model, api_token=hf_access_token)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            hf_service=hf_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
