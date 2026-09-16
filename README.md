# AIConversation

A conversation class for managing chats with AI models through an OpenAI compatible API.

It keeps the full history of a conversation, decides which part of it a model is allowed to see,
handles text / image / file messages, and persists everything to disk. It does **not** know how to call a model: it hands you the `messages` array and your program builds the request.

Tested against OpenAI-compatible endpoints including Azure OpenAI and local
[Ollama](https://ollama.com) / LM Studio servers.

---

## Design rule

**This module knows about conversations, not about how a model must be driven.**

It holds no model name, no temperature, no `max_tokens`, and never guesses what a model can accept.
Everything model-specific is passed in by the caller, who is the one that actually knows:

```python
body = {
    "model": "gpt-4o",                              # caller's business
    "messages": conv.get_memory_messages_for_ai(),  # conversation's business
    "temperature": 0.7,                             # caller's business
}
```

The payoff is that the same conversation can be sent to any model, in any order, without ever being
rewritten — including models with wildly different capabilities.

---

## Quick start

```python
import rom_const as rc
from AIConversation import AIConversation

conv = AIConversation("You are a helpful assistant.", max_memory=6)

conv.add_message(rc.ROLE_USER, "What is the capital of Italy?")
conv.add_message(rc.ROLE_ASSISTANT, "Rome.", model="gpt-4o")

messages = conv.get_memory_messages_for_ai()

conv.save_conversation("chat.jsonl")
later = AIConversation.from_file("chat.jsonl")
```

With the OpenAI SDK:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=conv.get_memory_messages_for_ai(),
    temperature=0.7,
)
conv.add_message(rc.ROLE_ASSISTANT, response.choices[0].message.content, model="gpt-4o")
```

---

## History vs memory

The central idea. The same messages, two views:

| | what it is | who sees it |
|---|---|---|
| **History** | everything that ever happened, in order, internal commands included | you, the log, the save file |
| **Memory** | the last `max_memory` messages + system prompt + sticky messages, stopping at the most recent `/NEWTOPIC` | the model |

Memory **never** contains internal messages — they are notes about the conversation, not conversation.

```python
conv.get_history_messages()          # AIMessage objects, the record
conv.get_memory_messages()           # AIMessage objects, what the model may see
conv.get_history_messages_for_ai()   # plain dicts, whole conversation
conv.get_memory_messages_for_ai()    # plain dicts, memory window   <-- the usual one
```

The `_for_ai` pair returns the `messages` array and nothing else — no model, no sampling settings.

### Controlling memory

```python
conv.set_max_memory_messages(10)          # resize the window (nothing is deleted)
conv.set_last_two_messages_stickiness(True)  # pin a Q&A pair so it survives the window
conv.new_topic()                          # memory restarts here; history keeps everything
conv.restart()                            # drop all messages except the system prompt
```

### `/NEWTOPIC` — restart memory without losing history

`conv.new_topic()` appends an internal `/NEWTOPIC` marker to the history. It is the **only** internal
message the class interprets: when memory is built, the class walks the history backwards from the
most recent message and **stops at the first `/NEWTOPIC` it meets**. Everything before the marker is
invisible to the model — including sticky messages — as if the conversation had just begun. Only
the system prompt is carried across.

```
history:  [system] [u1] [a1] [u2*] [a2] [/NEWTOPIC] [u3] [a3]        (* = sticky)
memory:   [system]                              [u3] [a3]
```

What it does and does not do:

- **History is untouched.** Nothing is deleted; `get_history_messages()`, `show_history()` and the
  save file still hold `u1`…`a2`. Compare with `restart()`, which really drops the messages.
- **The window restarts.** `max_memory` still applies, but it now only counts messages after the
  marker; messages before it do not occupy slots.
- **It beats sticky.** A pinned message before the marker stays out — pinning means "keep this in
  the window", not "keep this across topics". Pin it again (or restate it) after the `new_topic()`
  if it still matters.
- **The system prompt always survives**, since it is message 0 and is re-added whenever a marker
  is hit.
- **Tokens follow memory.** `get_memory_total_tokens()` reflects only the post-marker messages.

Use it when the user genuinely changes subject, or when a long conversation has drifted and the
model is being distracted by stale context: the record stays complete, the model gets a clean slate.

Any *other* text you pass to `add_comment()` — `"/MODEL:gpt-4o"`, `"switched to Anthropic"`,
whatever you like — is just an application note. It is stored verbatim in the history and shown by
`show_history()`, but the class never parses it and never sends it to a model. `/NEWTOPIC` is the
single exception.

### Sticky messages — pin what matters

The memory window only carries the last `max_memory` messages, so anything important eventually
falls out of what the model sees. Marking a message **sticky** pins it: it is pulled back into
memory even after it has left the window, so the model keeps seeing it **without** you having to
resend the whole conversation every time. Use it for facts stated once that must never fade —
"always answer in Italian", "the project is called Foo", the user's hard constraints.

```python
conv.add_message(rc.ROLE_USER, "The project is called Foo", is_sticky=True)  # sticky at creation
conv.set_nth_message_stickiness(3, True)          # pin an existing message
conv.set_last_message_stickiness(True)            # pin the latest one
conv.set_last_two_messages_stickiness(True)       # pin the last Q&A pair
```

The rules:

- The system prompt (message 0) is **always** sticky and cannot be unpinned.
- Sticky *expands* memory: pinned messages come **in addition to** the window, in chronological order.
- `/NEWTOPIC` beats sticky: a pinned message before the most recent topic marker stays out.
- A sticky *reasoning* message outside the window stays out too — `add_reasoning=True` only keeps
  reasoning that is already inside the window, it never widens the selection.
- Stickiness survives save/load, and `remove_non_sticky_messages()` spares pinned messages.

---

## Message types

`AIC_TYPE_TEXT`, `AIC_TYPE_IMAGE_URL`, `AIC_TYPE_FILE`, `AIC_TYPE_INTERNAL`, `AIC_TYPE_REASONING`.

```python
conv.add_message(rc.ROLE_USER, "What is in this picture?",
                 rc.AIC_TYPE_IMAGE_URL, "c:/pics/diagram.png")
```

Images and files may be an http URL, a `data:` URL, or a **local path** — a local path is read and
base64 encoded on the spot, and the stored url becomes the data URL.

### Which type each role may carry

Not every role accepts every type. The contract is declared once in
`rom_const.AIC_ALLOWED_TYPES_BY_ROLE` and enforced by the `AIMessage` constructor (on construction
**and** on load), which raises `ValueError` on any pair outside the table:

| role | text | image_url | file | internal | reasoning |
|-------------|:----:|:---------:|:----:|:--------:|:---------:|
| `system`    |  ✅  |           |      |          |           |
| `developer` |  ✅  |           |      |          |           |
| `user`      |  ✅  |    ✅     |  ✅  |          |           |
| `assistant` |  ✅  |    ✅     |  ✅  |          |    ✅     |
| `internal`  |      |           |      |    ✅    |           |

So an **internal** message carries only its command text — never an image — and **reasoning** is an
assistant-only kind of output. To change the policy, edit the table; the errors and this behaviour
follow automatically.

---

## Sending to a model that cannot handle attachments

Not every model accepts images, and an OpenAI-**compatible** API does not mean a **capable** model.
Ollama speaking OpenAI's dialect does not give `llama3.2` eyes:

```
HTTP 400 - "Multimodal data provided, but model does not support multimodal requests."
```

Tell the conversation what to leave out. The caller decides, because the caller is what knows which
model it is about to call:

```python
messages = conv.get_memory_messages_for_ai(
    strip_types={rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE}
)
```

Stripped messages are **not dropped** — the turn stays, keeping its caption, with the attachment
replaced by `[image omitted: not supported by the target model]`. Dropping the whole turn would
orphan the reply that referred to it.

**Stripping never touches stored history.** The same conversation still sends the full image to a
vision model afterwards:

```python
conv.get_memory_messages_for_ai(strip_types={rc.AIC_TYPE_IMAGE_URL})  # text-only model
conv.get_memory_messages_for_ai()                                     # vision model, image intact
```

Ollama publishes what each local model can do, which beats guessing:

```bash
curl -s http://localhost:11434/api/show -d '{"model":"gemma3:12b"}' | jq .capabilities
# ["completion","vision"]
```

---

## Persistence

```python
conv.save_conversation("chat.jsonl")
conv.load_conversation("chat.jsonl")
conv = AIConversation.from_file("chat.jsonl")
```

JSON Lines, one message per line. Only semantic fields are stored — never the request format — so
saved conversations are not tied to any endpoint's request shape and survive changes to it.

Saving is atomic: data goes to a `.tmp` file, is flushed and `fsync`'d, then renamed over the
target, with retries on transient I/O errors. Interrupting a save cannot corrupt the previous one.
A message that fails to load is reported and skipped rather than aborting the whole load.

---

## Display

```python
conv.show_memory()        # what the model currently sees
conv.show_history()       # everything, internal commands included
conv.show_memory_text()   # same, without the type/sticky/model header lines
```

Coloured by role, with data URLs abbreviated so a base64 image cannot flood your terminal.

---

## Token counters

```python
conv.get_total_tokens()
conv.get_memory_total_tokens()
conv.get_user_tokens(), conv.get_assistant_tokens()
```

**These are estimates for display, not a budget.** Text counts are exact (via `tiktoken`), but:

- **Image counts are rough.** They are derived from the encoded string length, while real vision
  pricing depends on pixel dimensions. An image supplied as a local path is measured on the length
  of its path, which is meaningless.
- **File contents are not counted** — only the caption is.

The authoritative number is the `usage` your API returns in its response.

---

## Files

| File | Purpose |
|---|---|
| `AIConversation.py` | `AIMessageContent`, `AIMessage`, `AIConversation` |
| `sample_usage.py` | a short, readable walkthrough of the API — start here |
| `test_AIConversation.py` | the self test / demonstration — every check, no network or API key needed |
| `rom_const.py` | roles, message types, the `/NEWTOPIC` marker, defaults |
| `rom_printing_utils.py` | coloured terminal output |

Requires Python 3.9+ and `tiktoken`.

```bash
python sample_usage.py           # a short tour of the API
python test_AIConversation.py    # the tests themselves
python AIConversation.py         # the module's main just calls run_tests()
```

The exit code of the test runs is 0 when every check passes, non-zero otherwise.

---

## Known limitations

- Image token estimates use byte size rather than pixel dimensions (see above).
- `AIMessage.is_internal()` reports the *type* (it is `True` for both `internal` and `reasoning`),
  but memory filtering does not use it: `get_memory_messages()` excludes reasoning by its **type**,
  so a reasoning message stays out of memory by default even though its role is `assistant`. Pass
  `add_reasoning=True` to feed it back to the model.
- `set_text()` refreshes that message's token estimate, but the conversation totals only follow
  after `recalculate_tokens()`.
- No `detail` parameter is emitted for images. It is an OpenAI-specific cost/quality knob and
  therefore a caller concern; add it to the content blocks yourself if you need it.

## License

MIT — see [LICENSE](LICENSE).
