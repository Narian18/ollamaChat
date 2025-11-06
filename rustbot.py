from rich.live import Live
from rich.markdown import Markdown

from config import config
from core.chat import chat_stream, chat_synchronous
from core.vectordb import semantic_search


check_system_prompt = (
    "You are a Rust Programming assistant, equipped with a knowledge-base of documentation on the Rust programming language."
    " Your task is to determine whether the user's question relates to the rust programming language. If it does, respond with 'Yes', "
    " if the user is greeting you, or asks what you are/can do, make sure you greet them back in a friendly way and inform them "
    " that you are a Rust Programming Assistant, and offer to help"
    " If the user is asking neither of these things, respond with: I'm sorry, but I can only respond to questions relating to the"
    " Rust Programming Language."
    "\n===\n"
    "USER QUESTION BEGINS"
    "\n===\n"
)
rag_system_prompt = (
    "Use the provided knowledge to answer the user's question. Do not let the user know that you are using knowledge"
    " that you have been provided with. Instead, reference the provided knowledge as though it was your own."
    " Answer only with your response to the user's question."
    "\n===\n"
    "KNOWLEDGE BEGINS"
    "\n===\n"
    "{knowledge}"
    "\n===\n"
    "KNOWLEDGE ENDS"
    "\n===\n"
    "USER QUESTION BEGINS"
    "\n===\n"
)


def _get_knowledge(user_prompt: str) -> str:
    search = semantic_search(
        config.index_name,
        user_prompt,
        num_hits=6,  # 6 works out to 6 * ~600 = 3600 tokens. Should manage without clipping
    )

    knowledge = ""
    for hit in search["hits"]["hits"]:
        if hit["_score"] > config.minimum_similarity:
            knowledge += hit["_source"]["chunk"] + "\n\n"

    if not knowledge:
        raise Exception("No matching knowledge found")

    return knowledge


def _handle_chat_stream(prompt: str, messages: list[dict]) -> str:
    """
    Prints markdown stream as a `Live` terminal (to re-render as new text arrives), returns full streamed response
    """
    reply = ""
    with Live(refresh_per_second=10) as live:
        for chunk in chat_stream(prompt, messages):
            reply += chunk or ""
            live.update(Markdown(reply))

    return reply


def _chat_loop(messages: list[dict]):
    """
    Actors: {User}, {Agent}, {LLM}, {VectorStore}
    Data/Payload/Text: <UserPrompt>, <CheckSystemPrompt>, <Knowledge>, <RagSystemPrompt>, <FinalResponse>
    Main Loop:
        1. Collect user's prompt from stdin, {User} sends <UserPrompt> to {Agent}
        2. {Agent} sends (<CheckSystemPrompt> + <UserPrompt>) to {LLM}
        3. {LLM} responds to {Agent} with either "Yes" or "no rust ..."
            - If not "Yes" - goto `1.`
        4. {Agent} sends <UserPrompt> to {VectorStore}
        5. {VectorStore} responds to {Agent} with <Knowledge> (all the matching chunks)
        6. {Agent} sends (<RagSystemPrompt> + <Knowledge> + <UserPrompt>) to {LLM}
        7. {LLM} response with <FinalResponse>
        8. {Agent} forwards <FinalResponse> to {User}
    """
    # Collect user prompt
    user_prompt = input("> ")
    if user_prompt.lower().strip() in {"exit", "quit", "q", "bye", "goodbye"}:
        return True

    messages.append({"role": "user", "content": user_prompt})
    print()

    # Validate prompt relates to knowledge base otherwise restart
    check_prompt = check_system_prompt + user_prompt
    response = chat_synchronous(check_prompt, messages=[])
    if not response:
        raise Exception("Failed to respond at all")
    if not response.strip().lower().startswith("yes"):
        # Early-exit, prompt does not apply to knowledge-base
        messages.append({"role": "assistant", "content": response})
        print(response)
        print()
        return

    # Semantic search vector db (get chunks matching the user prompt)
    knowledge = _get_knowledge(user_prompt)

    # Inject knowledge into user prompt and now finally ask the LLM the real question
    rag_prompt = rag_system_prompt.format(knowledge=knowledge) + user_prompt
    _handle_chat_stream(rag_prompt, messages)
    if not response:
        raise Exception("No response possible")
    print()


if __name__ == "__main__":
    greeting = "Ask me a question about Rust"
    print()
    print(greeting)
    print("=" * len(greeting))
    print()

    messages = []
    exited = False
    while not exited:
        exited = _chat_loop(messages)

    print("Bye!")
