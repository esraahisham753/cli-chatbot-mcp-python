import json
from typing import Optional, Literal, List
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[Tool]:
        """Gets all tools from the provided clients."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """Finds the first client that has the specified tool."""
        for client in clients:
            tools = await client.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                return client
        return None

    @classmethod
    async def execute_tool(
        cls, clients: dict[str, MCPClient], tool_name: str, tool_input: dict
    ) -> dict:
        """Executes a single tool request against the provided clients."""
        client = await cls._find_client_with_tool(list(clients.values()), tool_name)

        if not client:
            return {
                "tool": tool_name,
                "input": tool_input,
                "content": json.dumps({"error": "Could not find that tool"}),
                "is_error": True,
            }

        try:
            tool_output: CallToolResult | None = await client.call_tool(
                tool_name, tool_input
            )
            items = tool_output.content if tool_output else []
            content_list = [
                item.text for item in items if isinstance(item, TextContent)
            ]
            return {
                "tool": tool_name,
                "input": tool_input,
                "content": json.dumps(content_list),
                "is_error": bool(tool_output and tool_output.isError),
            }
        except Exception as e:
            error_message = f"Error executing tool '{tool_name}': {e}"
            print(error_message)
            return {
                "tool": tool_name,
                "input": tool_input,
                "content": json.dumps({"error": error_message}),
                "is_error": True,
            }

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success"] | Literal["error"],
    ) -> dict:
        """Builds a tool result part dictionary."""
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message
    ) -> List[dict]:
        """Executes a list of tool requests against the provided clients."""
        tool_requests = [
            block for block in message.content if block.type == "tool_use"
        ]
        tool_result_blocks: list[dict] = []
        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input

            result = await cls.execute_tool(clients, tool_name, tool_input)
            tool_result_part = cls._build_tool_result_part(
                tool_use_id,
                result["content"],
                "error" if result["is_error"] else "success",
            )

            tool_result_blocks.append(tool_result_part)
        return tool_result_blocks
