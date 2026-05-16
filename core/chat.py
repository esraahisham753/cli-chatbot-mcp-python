from core.huggingface import HuggingFace
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
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
        final_text_response = ""

        await self._process_query(query)

        # Hugging Face models don't support native tool calling like Claude
        # So we'll provide tool information in the system prompt instead
        system_prompt = await self._build_system_prompt()
        
        response = self.hf_service.chat(
            messages=self.messages,
            system=system_prompt,
        )

        self.hf_service.add_assistant_message(self.messages, response)
        final_text_response = self.hf_service.text_from_message(response)

        return final_text_response

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
        return base_prompt
