import os

from rich.live import Live
from rich.markdown import Markdown

from core.chat import MessageType, chat_stream, dump_chat
from config import config

error_response = "Something went wrong while answering your question. Maybe try again?"

def _init():
    try:
        os.mkdir("chats")
    except FileExistsError:
        pass


def _handle_chat_stream(prompt: str, messages: MessageType) -> str:
    """
    Prints markdown stream as a `Live` terminal (to re-render as new text arrives), returns full streamed response
    """
    reply = ""
    with Live(refresh_per_second=10) as live:
        for chunk in chat_stream(prompt, messages):
            reply += chunk or ""
            live.update(Markdown(reply))

    return reply


def _chat_loop(messages: MessageType):
    opener = f"Ask {config.model} anything!"
    print(f"{opener}\n{'=' * len(opener)}\n")

    while True:
        prompt = input("> ")
        if not prompt:
            print()
            continue

        print()
        reply = _handle_chat_stream(prompt, messages)

        if not reply:
            print(error_response)
            continue

        print()


if __name__ == "__main__":
    _init()
    messages = []

    try:
        _chat_loop(messages)
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nChat exited. Goodbye!\n")
        else:
            raise
    finally:
        dump_chat(messages)
