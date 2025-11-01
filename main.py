import os
from datetime import datetime

from ollama import chat
from rich.live import Live
from rich.markdown import Markdown

MODEL = "gemma3"
error_response = "Something went wrong while answering your question. Maybe try again?"

MessageType = list[dict[str, str]]


def _short_time() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _chat_stream(question: str, messages: MessageType):
    messages.append({"role": "user", "content": question})

    for chunk in chat(model=MODEL, messages=messages, stream=True):
        yield chunk.message.content


def _chat(question: str, messages: MessageType) -> str | None:
    messages.append({"role": "user", "content": question})

    response = chat(model="gemma3", messages=messages)
    response = response.message.content
    if not response:
        return
    else:
        messages.append({"role": "assistant", "content": response})

    return response


def _dump_chat(messages: MessageType):
    chat_summary = _chat(
        "Please summarise the preceding discussion into a max 6 word phrase. Respond with only the phrase, and nothing else.",
        messages,
    )
    if not chat_summary:
        raise Exception("Failed to save chat. Aborting")

    filename = chat_summary.title().replace(" ", "") + f"_{_short_time()}"

    with open(f"chats/{filename}.md", "w") as f:
        for message in messages:
            f.write(f"{message['role'].title()}: {message['content']}\n")


if __name__ == "__main__":
    os.mkdir("chats")

    opener = "Ask gemma anything!"
    print(f"{opener}\n{'=' * len(opener)}\n")

    messages = []
    try:
        while True:
            request = input("> ")
            if not request:
                print()
                continue

            print()
            reply = ""
            with Live(refresh_per_second=10) as live:
                for chunk in _chat_stream(request, messages):
                    reply += chunk or ""
                    live.update(Markdown(reply))

            if not reply:
                print(error_response)
                continue

            print()
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nChat exited. Goodbye!\n")
        else:
            raise
    finally:
        _dump_chat(messages)
