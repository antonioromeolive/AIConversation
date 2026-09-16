# Author:  Antonio Romeo
# Date:    2026-08-28
# Description: Minimal, readable walkthrough of the AIConversation public API.
#
# This is NOT a test - see test_AIConversation.py for the exhaustive self-test suite that
# exercises every edge case. This file is the opposite: the shortest useful tour of the API,
# meant to be read top to bottom. Run it directly:
#
#   python sample_usage.py
#
# MIT License (see the LICENSE file distributed with this project)

import os
import tempfile

import rom_const as rc
from AIConversation import AIConversation


def main() -> None:
    # 1. Build a conversation. The system prompt and the memory window (how many messages,
    #    excluding the system prompt, a model is allowed to see) are set once, up front.
    conv = AIConversation("You are a helpful, concise assistant.", max_memory=6)

    # 2. Add messages as the conversation happens.
    conv.add_message(rc.ROLE_USER, "What is the capital of Italy?")
    conv.add_message(rc.ROLE_ASSISTANT, "Rome.", model="gpt-4o")

    # 3. Pin something that must never fall out of memory, however long the chat gets.
    conv.add_message(rc.ROLE_USER, "From now on, always answer in Italian.", is_sticky=True)
    conv.add_message(rc.ROLE_ASSISTANT, "Certo! Da ora in poi rispondo in italiano.", model="gpt-4o")

    # 4. An attachment - a local path, an http url or a data: url all work the same way.
    conv.add_message(rc.ROLE_USER, "What is in this picture?",
                      rc.AIC_TYPE_IMAGE_URL, "https://example.com/diagram.png")
    conv.add_message(rc.ROLE_ASSISTANT, "Un diagramma con tre caselle collegate da frecce.", model="gpt-4o")

    # 5. add_comment() records something that HAPPENED (a parameter change, a note) rather than
    #    something that was SAID. It is stored as an internal message: it shows up in history and
    #    in the saved file, but it is never sent to a model and never counts toward memory or tokens.
    conv.add_comment("/MODEL:gpt-4o")
    in_memory  = any(m.get_type() == rc.AIC_TYPE_INTERNAL for m in conv.get_memory_messages())
    in_history = any(m.get_type() == rc.AIC_TYPE_INTERNAL for m in conv.get_history_messages())
    print(f"internal comment present in memory: {in_memory}, in history: {in_history}\n")

    # 6. get_memory_messages_for_ai() is what you hand an OpenAI-compatible client - notice it is
    #    NOT one message longer for the comment just added above:
    #
    #      response = client.chat.completions.create(
    #          model="gpt-4o", messages=conv.get_memory_messages_for_ai(), temperature=0.7,
    #      )
    messages = conv.get_memory_messages_for_ai()
    print(f"{len(messages)} messages ready to send to the model\n")

    # 7. HISTORY (everything, comment included) and MEMORY (what the model currently sees, comment
    #    excluded) are two views of the same messages - see the README for the full picture.
    print("-- memory (what the model currently sees; no internal comment) --")
    conv.show_memory()

    print("-- history (everything, unabridged, internal comment included) --")
    conv.show_history()

    # 8. new_topic() draws a line under the conversation: memory restarts here, nothing is
    #    deleted - history still has everything, and the pinned message above still doesn't
    #    make it back in, because /NEWTOPIC beats sticky.
    conv.new_topic()
    conv.add_message(rc.ROLE_USER, "Different subject: what is 2 + 2?")
    print("-- memory after new_topic() --")
    conv.show_memory()

    # 9. Token counters are estimates for on-screen display, not a budget - see the README.
    print(f"estimated tokens - total: {conv.get_total_tokens()}, "
          f"memory: {conv.get_memory_total_tokens()}")

    # 10. Persist to disk and load it back. JSON Lines, one message per line.
    saved_path = os.path.join(tempfile.gettempdir(), "aiconversation_sample.jsonl")
    conv.save_conversation(saved_path)
    reloaded = AIConversation.from_file(saved_path)
    print(f"\nsaved to {saved_path} and reloaded: {reloaded.count_all_messages()} messages")
    os.remove(saved_path)


if __name__ == '__main__':
    main()
