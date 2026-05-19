import json
import re

from core.huggingface import HuggingFace
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    TOOL_CALL_PATTERN = re.compile(
        r"@(?P<name>[A-Za-z_][\w-]*)\s*\(\s*(?P<input>\{.*?\})\s*\)",
        re.DOTALL,
    )

    def __init__(self, hf_service: HuggingFace, clients: dict[str, MCPClient]):
        self.hf_service: HuggingFace = hf_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
    ) -> str:
        await self._process_query(query)

        # Hugging Face models don't support native tool calling like Claude
        # So we'll provide tool information in the system prompt instead
        system_prompt = await self._build_system_prompt()

        for _ in range(3):
            response = self.hf_service.chat(
                messages=self.messages,
                system=system_prompt,
            )

            final_text_response = self.hf_service.text_from_message(response)
            tool_requests = self._extract_tool_requests(final_text_response)

            if not tool_requests:
                self.hf_service.add_assistant_message(self.messages, response)
                return final_text_response

            self.hf_service.add_assistant_message(self.messages, response)
            tool_results = await self._execute_tool_requests(tool_requests)
            self.messages.append(
                {
                    "role": "user",
                    "content": self._build_tool_result_prompt(tool_results),
                }
            )

        return final_text_response

    def _extract_tool_requests(self, text: str) -> list[dict]:
        """Extracts text-based tool calls like @read_doc({"doc_id": "report.pdf"})."""
        tool_requests = []

        for match in self.TOOL_CALL_PATTERN.finditer(text):
            try:
                tool_requests.append(
                    {
                        "name": match.group("name"),
                        "input": json.loads(match.group("input")),
                    }
                )
            except json.JSONDecodeError:
                continue

        return tool_requests

    async def _execute_tool_requests(self, tool_requests: list[dict]) -> list[dict]:
        results = []

        for tool_request in tool_requests:
            results.append(
                await ToolManager.execute_tool(
                    self.clients,
                    tool_request["name"],
                    tool_request["input"],
                )
            )

        return results

    def _build_tool_result_prompt(self, tool_results: list[dict]) -> str:
        return f"""
        The tool calls you requested have been executed. Use these results to answer the user's original question in your own words. Do not copy the tool result verbatim unless the user asks for exact text. Do not include another tool call
        unless another tool is still required.

        <tool_results>
        {json.dumps(tool_results, indent=2)}
        </tool_results>
        """

    async def _build_system_prompt(self) -> str:
        """Build a system prompt that includes available tools"""
        base_prompt = "You are a helpful assistant with access to the following tools:\n\n"
        
        tools = await ToolManager.get_all_tools(self.clients)
        
        for tool in tools:
            tool_desc = f"- {tool['name']}: {tool['description']}\n"
            if 'input_schema' in tool:
                tool_desc += f"  Input schema: {tool['input_schema']}\n"
            base_prompt += tool_desc
        
        base_prompt += "\nYou can reference these tools in your responses if needed."
        base_prompt += (
            '\n\nTo use a tool, write exactly one or more calls in this format: '
            '@tool_name({"argument": "value"}). After the tool result is provided, '
            "answer the user with the result instead of repeating the tool call."
        )
        return base_prompt
