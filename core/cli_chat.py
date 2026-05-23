from typing import List, Tuple
from mcp.types import Prompt, PromptMessage

from core.chat import Chat
from core.huggingface import HuggingFace
from core.tools import ToolManager
from mcp_client import MCPClient


class CliChat(Chat):
    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        hf_service: HuggingFace,
    ):
        super().__init__(clients=clients, hf_service=hf_service)

        self.doc_client: MCPClient = doc_client

    async def list_prompts(self) -> list[Prompt]:
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        return await self.doc_client.read_resource("docs://documents/")

    async def get_doc_content(self, doc_id: str) -> str:
        return await self.doc_client.read_resource(f"docs://documents/{doc_id}/")

    async def get_prompt(
        self, command: str, doc_id: str
    ) -> list[PromptMessage]:
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    async def run(self, query: str) -> str:
        words = query.split()
        if words and words[0] == "/format":
            return await self._run_format_command(words)

        return await super().run(query)

    async def _run_format_command(self, words: list[str]) -> str:
        if len(words) < 2:
            return "Command '/format' requires a document id."

        doc_id = words[1]
        doc_ids = await self.list_docs_ids()
        if doc_id not in doc_ids:
            return (
                f"Document '{doc_id}' was not found.\n\n"
                f"Available document ids: {', '.join(doc_ids)}"
            )

        content = await self.get_doc_content(doc_id)
        markdown = self._format_markdown(doc_id, content)

        result = await ToolManager.execute_tool(
            self.clients,
            "edit_doc",
            {
                "doc_id": doc_id,
                "old_string": content,
                "new_string": markdown,
            },
        )

        if result["is_error"]:
            return f"Could not format '{doc_id}': {result['content']}"

        self.messages.append({"role": "user", "content": f"/format {doc_id}"})
        self.messages.append({"role": "assistant", "content": markdown})

        return markdown

    def _format_markdown(self, doc_id: str, content: str) -> str:
        title = doc_id.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
        content = content.strip()

        if content.startswith("#"):
            return f"{content}\n"

        normalized_content = " ".join(content.split())
        sentences = [
            sentence.strip()
            for sentence in normalized_content.split(". ")
            if sentence.strip()
        ]

        if not sentences:
            return f"# {title}\n\n{content}\n"

        bullets = "\n".join(
            f"- {sentence if sentence.endswith('.') else sentence + '.'}"
            for sentence in sentences
        )
        return f"# {title}\n\n## Summary\n\n{bullets}\n"

    async def _extract_resources(self, query: str) -> str:
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[Tuple[str, str]] = []

        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned_docs.append((doc_id, content))

        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False

        words = query.split()
        doc_id = words[1] if len(words) > 1 else None
        if len(words) < 2:
            self.messages.append({
                "role": "user",
                "content": f"Command '{words[0]}' requires a document id.",
            })
            return True

        command = words[0].replace("/", "")
        doc_ids = await self.list_docs_ids()
        if doc_id not in doc_ids:
            self.messages.append({
                "role": "user",
                "content": (
                    f"Document '{doc_id}' was not found. "
                    f"Available document ids: {', '.join(doc_ids)}"
                ),
            })
            return True

        content = await self.get_doc_content(doc_id)

        messages = await self.doc_client.get_prompt(
            command, {"doc_id": doc_id}
        )

        self.messages.extend(convert_prompt_messages_to_message_params(messages))

        self.messages.append({
            "role": "user",
            "content": f"""
                Execute the command now.

                Document id: {doc_id}
                The application has already verified that this document exists.
                Do not say the document does not exist.
                Do not ask the user for the document id.

                Here is the exact current document content. If you edit the document,
                use the full text below as the old_string value.

                Current document content:
                {content}

                Required next action:
                Call edit_doc with doc_id "{doc_id}", old_string equal to the current
                document content above, and new_string equal to the markdown-formatted
                replacement.
            """
        })
       
        return True

    async def _process_query(self, query: str):
        if await self._process_command(query):
            return

        added_resources = await self._extract_resources(query)

        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note the user's query might contain references to documents like "@report.docx". The "@" is only
        included as a way of mentioning the doc. The actual name of the document would be "report.docx".
        If the document content is included in this prompt, you don't need to use an additional tool to read the document.
        Answer in your own words. Do not copy document text verbatim unless the user asks for the exact text.. Start with the exact information they need. 
        Don't refer to or mention the provided context in any way - just use it to inform your answer.
        """

        self.messages.append({"role": "user", "content": prompt})


def convert_prompt_message_to_message_param(
    prompt_message: "PromptMessage",
) -> dict:
    role = "user" if prompt_message.role == "user" else "assistant"

    content = prompt_message.content

    # Check if content is a dict-like object with a "type" field
    if isinstance(content, dict) or hasattr(content, "__dict__"):
        content_type = (
            content.get("type", None)
            if isinstance(content, dict)
            else getattr(content, "type", None)
        )
        if content_type == "text":
            content_text = (
                content.get("text", "")
                if isinstance(content, dict)
                else getattr(content, "text", "")
            )
            return {"role": role, "content": content_text}

    if isinstance(content, list):
        text_blocks = []
        for item in content:
            # Check if item is a dict-like object with a "type" field
            if isinstance(item, dict) or hasattr(item, "__dict__"):
                item_type = (
                    item.get("type", None)
                    if isinstance(item, dict)
                    else getattr(item, "type", None)
                )
                if item_type == "text":
                    item_text = (
                        item.get("text", "")
                        if isinstance(item, dict)
                        else getattr(item, "text", "")
                    )
                    text_blocks.append(item_text)

        if text_blocks:
            return {"role": role, "content": "\n".join(text_blocks)}

    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: List[PromptMessage],
) -> List[dict]:
    return [
        convert_prompt_message_to_message_param(msg) for msg in prompt_messages
    ]
