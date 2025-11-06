from datetime import datetime

from ollama import chat

from config import config


MessageType = list[dict[str, str]]


def _short_time() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def chat_stream(question: str, messages: MessageType):
    messages.append({"role": "user", "content": question})

    for chunk in chat(model=config.model, messages=messages, stream=True):
        yield chunk.message.content


def chat_synchronous(question: str, messages: MessageType) -> str | None:
    messages.append({"role": "user", "content": question})

    response = chat(model=config.model, messages=messages)
    response = response.message.content

    return response


def dump_chat(messages: MessageType):
    response = chat(
        "Please summarise the preceding discussion into a max 6 word phrase. Respond with only the phrase, and nothing else.",
        messages,
    )
    chat_summary = response.message.content
    if not chat_summary:
        raise Exception("Failed to save chat. Aborting")

    filename = chat_summary.title().replace(" ", "") + f"_{_short_time()}"

    with open(f"chats/{filename}.md", "w") as f:
        for message in messages:
            f.write(f"{message['role'].title()}: {message['content']}\n")


