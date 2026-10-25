"""Промпты и форматирование для ChatML."""
from llama_index.core import PromptTemplate

SYSTEM_PROMPT_TEMPLATE = """\
Ты — консультант магазина шин.
Отвечай ТОЛЬКО на основе контекста. Если ответа нет — скажи:
"К сожалению, в моей базе знаний нет информации по этому вопросу."

## КОНТЕКСТ:
{context_str}

## ВОПРОС КЛИЕНТА:
{query_str}

## ОТВЕТ:
"""


def build_qa_prompt() -> PromptTemplate:
    return PromptTemplate(SYSTEM_PROMPT_TEMPLATE)


def messages_to_prompt(messages) -> str:
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
    if not prompt.endswith("<|im_start|>assistant\n"):
        prompt += "<|im_start|>assistant\n"
    return prompt


def completion_to_prompt(completion: str) -> str:
    return f"<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"