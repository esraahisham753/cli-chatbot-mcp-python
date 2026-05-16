from huggingface_hub import InferenceClient
import os


class HuggingFaceMessage:
    """Wrapper to mimic Anthropic Message format for compatibility"""
    def __init__(self, content: str):
        self.content = [{"type": "text", "text": content}]
        self.stop_reason = "end_turn"


class HuggingFace:
    def __init__(self, model: str, api_token: str = ""):
        self.model = model
        self.client = InferenceClient(
            model=model,
            token=api_token or os.getenv("HF_ACCESS_TOKEN", "")
        )

    def add_user_message(self, messages: list, message):
        user_message = {
            "role": "user",
            "content": message.content
            if hasattr(message, 'content') and isinstance(message.content, str)
            else (message if isinstance(message, str) else str(message)),
        }
        messages.append(user_message)

    def add_assistant_message(self, messages: list, message):
        content = message.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        
        assistant_message = {
            "role": "assistant",
            "content": content if isinstance(content, str) else str(message),
        }
        messages.append(assistant_message)

    def text_from_message(self, message):
        if isinstance(message.content, list):
            return "\n".join([
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in message.content
            ])
        return message.content

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ):
        # Convert messages to format expected by Hugging Face
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Build the chat prompt
        system_prompt = system if system else ""
        
        try:
            response_text = self.client.text_generation(
                prompt=self._build_prompt(formatted_messages, system_prompt),
                max_new_tokens=1000,
                temperature=temperature,
                stop_sequences=stop_sequences or [],
            )
        except Exception as e:
            # Fallback for inference API
            response_text = self._fallback_chat(formatted_messages, system_prompt, temperature)
        
        return HuggingFaceMessage(response_text)

    def _build_prompt(self, messages: list, system: str) -> str:
        """Build a prompt string from messages"""
        prompt = ""
        
        if system:
            prompt += f"System: {system}\n\n"
        
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        
        prompt += "Assistant: "
        return prompt

    def _fallback_chat(self, messages: list, system: str, temperature: float) -> str:
        """Fallback chat using chat_completion endpoint if available"""
        try:
            chat_messages = []
            if system:
                chat_messages.append({"role": "system", "content": system})
            chat_messages.extend(messages)
            
            response = self.client.chat_completion(
                messages=chat_messages,
                max_tokens=1000,
                temperature=temperature,
            )
            
            if isinstance(response, dict):
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return str(response)
        except Exception:
            return "Unable to generate response"
