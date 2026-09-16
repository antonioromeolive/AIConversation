# Author:  Antonio Romeo
# Date:    2026-07-23
# Description: Self test / demonstration for AIConversation.py.
#
# Exercises the public API of AIMessage / AIConversation and shows the different views of the
# same messages. Nothing here needs a network or an API key. Run it directly:
#
#   python test_AIConversation.py
#
# or through the module it tests (python AIConversation.py), which simply calls run_tests().
#
# MIT License (see the LICENSE file distributed with this project)

import json
import os
import tempfile

import rom_const as rc
from rom_printing_utils import printColor, printColorSameLine
from AIConversation import AIMessage, AIConversation


def run_tests() -> bool:
    """ Run every check, print a verdict, and return True when they all passed. """
    _failures: list[str] = []

    #a real 1x1 red PNG, so the image path is exercised without needing a file on disk
    DEMO_PNG:str  = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                     "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    DEMO_FILE:str = "data:text/plain;base64,aGVsbG8gZnJvbSBhIHRlc3QgZmlsZQ=="   #"hello from a test file"


    def _section(title:str) -> None:
        printColor("\n" + "=" * 78, "bold_cyan")
        printColor(f" {title}", "bold_cyan")
        printColor("=" * 78, "bold_cyan")


    def _check(label:str, condition:bool, extra:str = "") -> None:
        """ Assert-with-a-face: report and keep going, so one failure does not hide the rest. """
        if condition:
            printColorSameLine("  ok   ", "light_green")
        else:
            printColorSameLine("  FAIL ", "light_red")
            _failures.append(label)
        print(f"{label} {extra}")


    #small probes for "does this input get refused?". They return True when the call SUCCEEDS,
    #so a rejection test reads as `not _accepts(...)`.
    def _accepts(role:"str | None", text:str, msg_type:str, image_url:"str | None" = None,
                 file_url:"str | None" = None) -> bool:
        """ Build a message in isolation: probing validation must not mutate the conversation. """
        try:
            AIMessage(role, text, msg_type, image_url, content_file_url=file_url)
            return True
        except ValueError:
            return False

    def _reject_reason(role:"str | None", text:str, msg_type:str, image_url:"str | None" = None,
                       file_url:"str | None" = None) -> "str | None":
        """ Like _accepts, but returns the refusal TEXT (None when the message builds). Lets a test
            confirm a bad message fails with a clean, informative ValueError - not a bare crash, and
            not some unrelated error from deeper in the constructor. """
        try:
            AIMessage(role, text, msg_type, image_url, content_file_url=file_url)
            return None
        except ValueError as exc:
            return str(exc)

    #every content type, and the content each one needs to build once its (role, type) pair is legal.
    #used to sweep the whole matrix: a legal pair must build, so its required content must be present,
    #otherwise a missing-content failure would masquerade as a contract failure.
    _ALL_TYPES = [rc.AIC_TYPE_TEXT, rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE,
                  rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING]
    _ALL_ROLES = [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER, rc.ROLE_USER, rc.ROLE_ASSISTANT, rc.ROLE_INTERNAL]

    def _accepts_pair(role:str, msg_type:str) -> bool:
        """ Can this (role, type) pair build when its required content IS supplied? Isolates the
            contract from the per-type content rules. """
        return _accepts(role, "content",  msg_type,
                        image_url = "http://x/y.png" if msg_type == rc.AIC_TYPE_IMAGE_URL else None,
                        file_url  = "http://x/y.pdf" if msg_type == rc.AIC_TYPE_FILE else None)


    def _reads(conversation:AIConversation, index:int) -> bool:
        try:
            conversation.get_ith_message(index)
            return True
        except ValueError:
            return False


    def _resizes(conversation:AIConversation, size:int) -> bool:
        try:
            conversation.set_max_memory_messages(size)
            return True
        except ValueError:
            return False


    def _removes(conversation:AIConversation, how_many:int) -> bool:
        try:
            conversation.remove_messages(how_many)
            return True
        except ValueError:
            return False


    # ---------------------------------------------------------------- build & system prompt
    _section("1. Building a conversation")

    conv = AIConversation("You are a terse test assistant.", max_memory=4)
    print(f"  system prompt   : {conv.get_system_prompt()}")
    print(f"  max memory      : {conv.get_max_memory_messages()}")
    _check("system prompt stored", conv.get_system_prompt().startswith("You are a terse"))
    _check("conversation starts with 1 message", conv.count_all_messages() == 1)

    conv.change_system_message("You are a terse test assistant. Answer in few words.")
    _check("change_system_message replaces message 0", "few words" in conv.get_system_prompt())
    _check("and does not append", conv.count_all_messages() == 1)


    # ---------------------------------------------------------------- adding every message type
    _section("2. Adding messages of every type")

    conv.add_message(rc.ROLE_USER, "What is the capital of Italy?")
    conv.add_message(rc.ROLE_ASSISTANT, "Thinking about it...", model="gpt-4o", msg_type=rc.AIC_TYPE_REASONING)
    conv.add_message(rc.ROLE_ASSISTANT, "Rome.", model="gpt-4o")
    conv.add_comment("/MODEL:gemma3:12b")                                  #application note: logged, never sent to AI
    conv.add_message(rc.ROLE_USER, "What is in this picture?", rc.AIC_TYPE_IMAGE_URL, DEMO_PNG)
    conv.add_message(rc.ROLE_ASSISTANT, "A red pixel.", model="gemma3:12b")
    conv.add_message(rc.ROLE_USER, "Summarise this document", rc.AIC_TYPE_FILE, file_url=DEMO_FILE)
    conv.add_message(rc.ROLE_ASSISTANT, "It says hello.", rc.AIC_TYPE_REASONING)

    for msg_type, count in sorted(conv.count_memory_types().items()):
        print(f"  memory holds {count} x {msg_type}")

    #8 messages added on top of the system prompt. Every accepted (role, type) combination is
    #exercised here; the exhaustive REFUSAL testing lives in section 2b, just below.
    _check("all messages recorded", conv.count_all_messages() == 9, f"({conv.count_all_messages()})")
    _check("a plain valid message is accepted", _accepts(rc.ROLE_USER, "hi", rc.AIC_TYPE_TEXT))


    # ---------------------------------------------- wrong messages are refused, exhaustively
    #an image given as a local file whose type cannot be guessed as an image is stored as a
    #"data:<other mime>" url. That url must NOT be mistaken for a path again by set_text()
    odd_image = AIMessage(rc.ROLE_USER, "caption", rc.AIC_TYPE_IMAGE_URL, DEMO_FILE)
    odd_image.set_text("new caption")
    _check("a non-image data: url survives set_text() untouched",
           odd_image.get_content().get_content()[0]["image_url"]["url"] == DEMO_FILE)
    _check("an image with a None caption builds with an empty caption",
           AIMessage(rc.ROLE_USER, None, rc.AIC_TYPE_IMAGE_URL, DEMO_PNG).get_text() == "")


    _section("2b. Every wrong (role, type) message is refused")

    #Sweep the ENTIRE 5x5 grid against the declared contract. For each pair, supply the content that
    #type needs, so the only thing under test is whether the pair itself is allowed. The table is the
    #oracle: this proves the constructor actually consults it, in every cell, not just the few probed
    #by hand above. A regression that dropped the pair check would light up here as accepted-when-it-
    #should-reject.
    _grid_ok = True
    for _role in _ALL_ROLES:
        for _mtype in _ALL_TYPES:
            _should = _mtype in rc.AIC_ALLOWED_TYPES_BY_ROLE[_role]
            if _accepts_pair(_role, _mtype) != _should:
                _grid_ok = False
                printColor(f"    grid mismatch: {_role} + {_mtype} "
                           f"(expected {'accept' if _should else 'reject'})", "light_red")
    _check("every (role, type) pair matches the contract table", _grid_ok)

    #Named illegal pairs, spelled out so the INTENT is asserted independently of the table above:
    #if someone loosens the table by mistake, these still pin the design down.
    _forbidden = [
        (rc.ROLE_SYSTEM,    rc.AIC_TYPE_IMAGE_URL),   #a system prompt is text, not a picture
        (rc.ROLE_SYSTEM,    rc.AIC_TYPE_FILE),
        (rc.ROLE_SYSTEM,    rc.AIC_TYPE_REASONING),   #the system does not "reason"
        (rc.ROLE_SYSTEM,    rc.AIC_TYPE_INTERNAL),
        (rc.ROLE_DEVELOPER, rc.AIC_TYPE_IMAGE_URL),
        (rc.ROLE_USER,      rc.AIC_TYPE_REASONING),   #reasoning is model output, not user input
        (rc.ROLE_USER,      rc.AIC_TYPE_INTERNAL),    #a command must not wear a user's face
        (rc.ROLE_ASSISTANT, rc.AIC_TYPE_INTERNAL),
        (rc.ROLE_INTERNAL,  rc.AIC_TYPE_TEXT),        #an internal message is ONLY its command
        (rc.ROLE_INTERNAL,  rc.AIC_TYPE_IMAGE_URL),
        (rc.ROLE_INTERNAL,  rc.AIC_TYPE_FILE),
        (rc.ROLE_INTERNAL,  rc.AIC_TYPE_REASONING),
    ]
    for _role, _mtype in _forbidden:
        _check(f"{_role} + {_mtype} refused", not _accepts_pair(_role, _mtype))

    #A refused pair must fail cleanly and say WHY: a ValueError that names the offending role and type.
    _reason = _reject_reason(rc.ROLE_INTERNAL, "look", rc.AIC_TYPE_IMAGE_URL, "http://x/y.png")
    _check("a forbidden pair reports a reason", _reason is not None)
    _check("the reason names the role and the type",
           _reason is not None and rc.ROLE_INTERNAL in _reason and rc.AIC_TYPE_IMAGE_URL in _reason)

    #The other failure family: a legal pair whose REQUIRED content is missing. Distinct from a bad
    #pair, and each must still be refused rather than stored half-built.
    _check("text with no text refused",        not _accepts(rc.ROLE_USER, "", rc.AIC_TYPE_TEXT))
    _check("image with no url refused",         not _accepts(rc.ROLE_USER, "cap", rc.AIC_TYPE_IMAGE_URL))
    _check("file with no url refused",          not _accepts(rc.ROLE_USER, "cap", rc.AIC_TYPE_FILE))
    _check("internal with no command refused",  not _accepts(rc.ROLE_INTERNAL, "", rc.AIC_TYPE_INTERNAL))
    _check("reasoning with no text refused",    not _accepts(rc.ROLE_ASSISTANT, "", rc.AIC_TYPE_REASONING))

    #Unknown role and unknown type stay refused, and a None role must not crash the error path itself.
    _check("unknown role refused",  not _accepts("wizard", "hi", rc.AIC_TYPE_TEXT))
    _check("unknown type refused",  not _accepts(rc.ROLE_USER, "hi", "hologram"))
    _check("role=None refused as a ValueError", not _accepts(None, "hi", rc.AIC_TYPE_TEXT))

    #add_message is the public door and must refuse the same wrong messages the constructor does.
    _add_refused = False
    try:
        conv.add_message(rc.ROLE_INTERNAL, "look", rc.AIC_TYPE_IMAGE_URL, image_url="http://x/y.png")
    except ValueError:
        _add_refused = True
    _check("add_message refuses a forbidden pair too", _add_refused)
    _check("...and the conversation was not mutated by the failed add", conv.count_all_messages() == 9)

    #set_role() is the back door: it must not let a message slip into a pair the constructor refuses.
    #Swept over the whole table, then the two ways it can fail spelled out.
    _reroled_ok = True
    for _from in _ALL_ROLES:
        for _mtype in _ALL_TYPES:
            if _mtype not in rc.AIC_ALLOWED_TYPES_BY_ROLE[_from]:
                continue
            for _to in _ALL_ROLES:
                _probe = AIMessage(_from, "content", _mtype,
                                   "http://x/y.png" if _mtype == rc.AIC_TYPE_IMAGE_URL else None,
                                   content_file_url = "http://x/y.pdf" if _mtype == rc.AIC_TYPE_FILE else None)
                _should = _mtype in rc.AIC_ALLOWED_TYPES_BY_ROLE[_to]
                try:
                    _probe.set_role(_to)
                    _did = True
                except ValueError:
                    _did = False
                if _did != _should or _probe.get_role() != (_to if _should else _from):
                    _reroled_ok = False
                    printColor(f"    set_role mismatch: {_from} + {_mtype} -> {_to} "
                               f"(expected {'accept' if _should else 'reject'})", "light_red")
    _check("set_role() honours the contract table for every (role, type, new role)", _reroled_ok)

    _picture = AIMessage(rc.ROLE_USER, "look", rc.AIC_TYPE_IMAGE_URL, "http://x/y.png")
    _picture.set_role(rc.ROLE_ASSISTANT)
    _check("set_role() to a role that may carry the content works", _picture.get_role() == rc.ROLE_ASSISTANT)
    _set_role_refused = False
    try:
        _picture.set_role(rc.ROLE_INTERNAL)
    except ValueError as exc:
        _set_role_refused = rc.ROLE_INTERNAL in str(exc) and rc.AIC_TYPE_IMAGE_URL in str(exc)
    _check("set_role() refuses a forbidden pair and names it", _set_role_refused)
    _check("...leaving the role untouched", _picture.get_role() == rc.ROLE_ASSISTANT)
    _unknown_refused = False
    try:
        _picture.set_role("wizard")
    except ValueError:
        _unknown_refused = True
    _check("set_role() refuses an unknown role", _unknown_refused)


    # ---------------------------------------------------------------- the two views
    _section("3. HISTORY vs MEMORY - the same messages, two views")

    printColor("\n  -- history (everything, internal commands included) --", "bold_yellow")
    conv.show_history()

    printColor("  -- memory (only what a model may see) --", "bold_yellow")
    conv.show_memory()

    printColor("  -- memory again, text only (show_memory_text) --", "bold_yellow")
    printed_memory:int = conv.show_memory_text()

    printColor("  -- history, text only (show_history_text) --", "bold_yellow")
    printed_history:int = conv.show_history_text()

    _check("show_memory_text printed something", printed_memory > 0)
    _check("history prints more than memory", printed_history > printed_memory)

    #print is the engine behind every show_* wrapper
    _check("print with add_internal=True prints more than without",
           conv.print(memory_only=False, add_internal=True, text_only=True) >
           conv.print(memory_only=False, add_internal=False, text_only=True))

    history = conv.get_history_messages()
    memory  = conv.get_memory_messages()
    print(f"  history: {len(history)} messages, memory: {len(memory)} messages")

    _check("history keeps internal commands",
           any(m.get_role() == rc.ROLE_INTERNAL for m in history))
    _check("memory never contains internal commands",
           all(m.get_role() != rc.ROLE_INTERNAL for m in memory))
    _check("history can hide internals on request",
           all(m.get_role() != rc.ROLE_INTERNAL for m in conv.get_history_messages(ignore_internals=True)))
    _check("memory is bounded by max_memory (+ system prompt)",
           len(memory) <= conv.get_max_memory_messages() + 1, f"({len(memory)})")
    #the newest message here is a REASONING message, which memory deliberately leaves out,
    #so memory ends at the most recent message a model is actually allowed to see
    last_visible = [m for m in conv.get_history_messages()
                    if m.get_role() != rc.ROLE_INTERNAL and m.get_type() != rc.AIC_TYPE_REASONING][-1]
    _check("memory ends at the newest message a model may see",
           memory[-1].get_text() == last_visible.get_text())
    _check("and that is NOT the newest message overall",
           memory[-1].get_text() != conv.get_last_message_text())


    # ---------------------------------------------------------------- AI facing arrays
    _section("4. The AI facing arrays")

    ai_memory  = conv.get_memory_messages_for_ai()
    ai_history = conv.get_history_messages_for_ai()

    print(f"  memory  array roles: {[m['role'] for m in ai_memory]}")
    print(f"  history array roles: {[m['role'] for m in ai_history]}")

    _check("plain dicts, not AIMessage objects", all(isinstance(m, dict) for m in ai_memory))
    _check("only role and content cross the boundary",
           all(set(m.keys()) == {"role", "content"} for m in ai_memory))
    _check("no generation settings leak in",
           all(k not in m for m in ai_memory for k in ("model", "temperature", "max_tokens")))
    _check("no internal messages reach the model",
           all(m["role"] != rc.ROLE_INTERNAL for m in ai_memory + ai_history))
    _check("history array is the longer one", len(ai_history) > len(ai_memory))

    printColor("\n  the calling program assembles the request:", "gray")
    print('    body = {"model": m, "messages": conv.get_memory_messages_for_ai(), "temperature": 0.7}')


    # ---------------------------------------------------------------- history-for-AI reasoning flag
    _section("4b. get_history_messages_for_ai() honours add_reasoning too")

    hist_no_reasoning   = conv.get_history_messages_for_ai()
    hist_with_reasoning = conv.get_history_messages_for_ai(add_reasoning=True)
    _types_no   = {b["type"] for m in hist_no_reasoning for b in m["content"]}
    _types_with = {b["type"] for m in hist_with_reasoning for b in m["content"]}

    _check("reasoning is excluded from the history-for-AI array by default",
           rc.AIC_TYPE_REASONING not in _types_no)
    _check("add_reasoning=True brings reasoning into the history-for-AI array too",
           rc.AIC_TYPE_REASONING in _types_with)
    _check("bringing reasoning in makes the array longer, never shorter",
           len(hist_with_reasoning) >= len(hist_no_reasoning))
    _check("still no internal messages, whatever add_reasoning says",
           all(m["role"] != rc.ROLE_INTERNAL for m in hist_with_reasoning))


    # ---------------------------------------------------------------- stripping attachments
    _section("5. Stripping attachments for a model that cannot accept them")

    stripped = conv.get_memory_messages_for_ai(strip_types={rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE})
    blocks_before = sorted({b["type"] for m in ai_memory for b in m["content"]})
    blocks_after  = sorted({b["type"] for m in stripped for b in m["content"]})
    print(f"  content blocks before: {blocks_before}")
    print(f"  content blocks after : {blocks_after}")

    _check("attachments are gone",
           rc.AIC_TYPE_IMAGE_URL not in blocks_after and rc.AIC_TYPE_FILE not in blocks_after)
    _check("but the turns are kept", len(stripped) == len(ai_memory))
    _check("the caption survives",
           any("What is in this picture?" in b.get("text", "") for m in stripped for b in m["content"]))
    _check("a placeholder explains the gap",
           any(rc.AIC_PLACEHOLDER_IMAGE in b.get("text", "") for m in stripped for b in m["content"]))
    #the same degradation, applied by hand to any list the caller happens to hold
    by_hand = AIConversation.strip_message_types(conv.get_memory_messages(), {rc.AIC_TYPE_IMAGE_URL})
    _check("strip_message_types works on any list of messages",
           all(m.get_type() != rc.AIC_TYPE_IMAGE_URL for m in by_hand))

    _check("STORED HISTORY IS UNTOUCHED",
           any(m.get_type() == rc.AIC_TYPE_IMAGE_URL for m in conv.get_history_messages()))
    _check("so a vision model still gets the image afterwards",
           rc.AIC_TYPE_IMAGE_URL in {b["type"] for m in conv.get_memory_messages_for_ai() for b in m["content"]})


    # ---------------------------------------------------------------- reasoning
    _section("5b. Reasoning messages are kept out of memory unless asked for")

    default_types  = sorted(conv.count_memory_types().keys())
    with_reasoning = sorted(conv.count_memory_types(add_reasoning=True).keys())
    print(f"  memory types, default            : {default_types}")
    print(f"  memory types, add_reasoning=True : {with_reasoning}")

    _check("reasoning is NOT in memory by default", rc.AIC_TYPE_REASONING not in default_types)
    _check("reasoning is in history regardless",
           any(m.get_type() == rc.AIC_TYPE_REASONING for m in conv.get_history_messages()))
    _check("add_reasoning=True brings it back", rc.AIC_TYPE_REASONING in with_reasoning)
    _check("the AI array follows the same rule",
           rc.AIC_TYPE_REASONING not in {b["type"] for m in conv.get_memory_messages_for_ai()
                                         for b in m["content"]})
    _check("...and follows the flag too",
           rc.AIC_TYPE_REASONING in {b["type"] for m in conv.get_memory_messages_for_ai(add_reasoning=True)
                                     for b in m["content"]})

    #excluding a message must not waste a slot in the window
    _check("a hidden reasoning message does not consume a window slot",
           len(conv.get_memory_messages()) >= len(conv.get_memory_messages(add_reasoning=True)))

    #reasoning is model output but never part of memory, so like INTERNAL it is costed at 0
    _check("a reasoning message costs 0 tokens",
           all(m.get_estimated_tokens() == 0 for m in conv.get_history_messages()
               if m.get_type() == rc.AIC_TYPE_REASONING))


    # ---------------------------------------------------------------- sticky messages
    _section("6. Sticky messages survive the memory window")

    conv.set_nth_message_stickiness(1, True)          #the very first question
    print(f"  sticky messages: {conv.count_sticky_messages()}, non sticky: {conv.count_non_sticky_messages()}")

    for i in range(3):
        conv.add_message(rc.ROLE_USER, f"filler question {i}")
        conv.add_message(rc.ROLE_ASSISTANT, f"filler answer {i}")

    with_sticky    = conv.get_memory_messages(add_sticky=True)
    without_sticky = conv.get_memory_messages(add_sticky=False)
    pinned_in      = any("capital of Italy" in m.get_text() for m in with_sticky)
    pinned_out     = any("capital of Italy" in m.get_text() for m in without_sticky)
    print(f"  pinned question inside window with add_sticky=True : {pinned_in}")
    print(f"  ... with add_sticky=False                          : {pinned_out}")

    _check("sticky message pulled back into memory", pinned_in)
    _check("and left out when not requested", not pinned_out)
    _check("set_nth_message_stickiness(0) cannot unstick the system prompt",
           conv.get_ith_message(0).is_sticky())

    conv.set_last_message_stickiness(True)
    _check("set_last_message_stickiness pins the latest message", conv.get_ith_message(-1).is_sticky())

    conv.set_last_two_messages_stickiness(True)
    _check("set_last_two_messages_stickiness pins a Q&A pair",
           conv.get_ith_message(-1).is_sticky() and conv.get_ith_message(-2).is_sticky())

    #sticky at CREATION time, not only via the set_* methods afterwards
    conv.add_message(rc.ROLE_USER, "PIN THIS: the project is called Foo", is_sticky=True)
    _check("add_message(is_sticky=True) builds the message already pinned",
           conv.get_ith_message(-1).is_sticky())

    #a sticky REASONING message is different: add_sticky reaches outside the window for text,
    #but never for reasoning - add_reasoning only keeps reasoning already inside the window
    conv.add_message(rc.ROLE_ASSISTANT, "pinned thinking", rc.AIC_TYPE_REASONING)
    conv.set_last_message_stickiness(True)

    for i in range(5):                     #push both pinned messages well outside the window of 4
        conv.add_message(rc.ROLE_USER, f"more filler {i}")
        conv.add_message(rc.ROLE_ASSISTANT, f"more filler answer {i}")

    _check("a sticky TEXT message outside the window is pulled back",
           any("PIN THIS" in m.get_text() for m in conv.get_memory_messages(add_sticky=True)))
    _check("a sticky REASONING message outside the window stays out",
           all("pinned thinking" not in m.get_text()
               for m in conv.get_memory_messages(add_sticky=True, add_reasoning=True)))

    #pulled-back stickies keep their place in time: memory must stay chronological
    _mem_view  = conv.get_memory_messages(add_sticky=True)
    _hist_view = conv.get_history_messages()
    _positions = [_hist_view.index(m) for m in _mem_view]
    _check("memory stays in chronological order with pulled-back stickies",
           _positions == sorted(_positions))


    # ---------------------------------------------------------------- sticky x memory combinations
    _section("6b. Sticky messages x memory-window flags - every combination")

    #isolated, small, fully-controlled conversations - `conv` by this point carries too much
    #accumulated state (filler, pins, a reasoning message) to reason about precisely
    combo = AIConversation("System prompt for combo tests.", max_memory=3)
    combo.add_message(rc.ROLE_USER, "pinned early message", is_sticky=True)
    combo.add_message(rc.ROLE_USER, "early filler, not sticky")
    combo.add_message(rc.ROLE_USER, "filler B")
    combo.add_message(rc.ROLE_ASSISTANT, "filler C")
    combo.add_message(rc.ROLE_USER, "filler D")

    #window is 3: the natural window is [filler B, filler C, filler D]. "pinned early message" and
    #"early filler" have both fallen outside it; only the sticky one can be pulled back.
    mem_with_sticky = {m.get_text() for m in combo.get_memory_messages(add_sticky=True)}
    mem_no_sticky    = {m.get_text() for m in combo.get_memory_messages(add_sticky=False)}
    system_prompt    = {"System prompt for combo tests."}
    natural_window   = {"filler B", "filler C", "filler D"}

    _check("add_sticky=True pulls the pinned message back from outside the window",
           mem_with_sticky == system_prompt | natural_window | {"pinned early message"}, f"{mem_with_sticky}")
    _check("add_sticky=False leaves memory at exactly the system prompt plus the natural window",
           mem_no_sticky == system_prompt | natural_window, f"{mem_no_sticky}")
    _check("a non-sticky message pushed outside the window never comes back",
           "early filler, not sticky" not in mem_with_sticky)

    #a message that is sticky AND still inside the natural window must not be dropped by
    #add_sticky=False - the flag only ever PULLS messages in, it never filters the window itself
    combo2 = AIConversation("System prompt.", max_memory=3)
    combo2.add_message(rc.ROLE_USER, "sticky and still current", is_sticky=True)
    combo2.add_message(rc.ROLE_ASSISTANT, "reply")
    _check("a sticky message still inside the window survives add_sticky=False",
           any("sticky and still current" in m.get_text()
               for m in combo2.get_memory_messages(add_sticky=False)))

    #no sticky messages anywhere: add_sticky must be a true no-op
    combo3 = AIConversation("System prompt.", max_memory=2)
    for i in range(4):
        combo3.add_message(rc.ROLE_USER, f"q{i}")
        combo3.add_message(rc.ROLE_ASSISTANT, f"a{i}")
    _check("add_sticky is a no-op when nothing in the conversation is sticky",
           [m.get_text() for m in combo3.get_memory_messages(add_sticky=True)] ==
           [m.get_text() for m in combo3.get_memory_messages(add_sticky=False)])

    #ROLE_INTERNAL is never memory-eligible, sticky or not, inside the window or out
    combo4 = AIConversation("System prompt.", max_memory=5)
    combo4.add_comment("a comment that happens to be marked sticky")
    combo4.set_last_message_stickiness(True)
    _check("a sticky INTERNAL message never enters memory, even fresh inside the window",
           all(m.get_role() != rc.ROLE_INTERNAL for m in combo4.get_memory_messages(add_sticky=True)))

    #a sticky REASONING message: add_sticky reaches outside the window for TEXT, never for
    #REASONING - the selection hard-codes add_reasoning=False for anything outside the window
    combo5 = AIConversation("System prompt.", max_memory=2)
    combo5.add_message(rc.ROLE_ASSISTANT, "sticky reasoning", rc.AIC_TYPE_REASONING)
    combo5.set_last_message_stickiness(True)
    for i in range(3):
        combo5.add_message(rc.ROLE_USER, f"q{i}")
        combo5.add_message(rc.ROLE_ASSISTANT, f"a{i}")
    _check("a sticky reasoning message pushed outside the window stays out even with add_reasoning=True",
           all("sticky reasoning" != m.get_text()
               for m in combo5.get_memory_messages(add_sticky=True, add_reasoning=True)))

    #the tightest legal window (max_memory=1) still behaves
    combo6 = AIConversation("System prompt.", max_memory=1)
    combo6.add_message(rc.ROLE_USER, "does not fit")
    combo6.add_message(rc.ROLE_ASSISTANT, "only this fits")
    mem6 = combo6.get_memory_messages()
    _check("max_memory=1 keeps exactly the system prompt plus the single newest message",
           [m.get_text() for m in mem6] == ["System prompt.", "only this fits"])

    #add_comment() stores ANY text verbatim now - it used to silently drop unrecognised commands
    combo.add_comment("/NOT_A_REGISTERED_COMMAND free text")
    _check("add_comment keeps text the old command whitelist would have silently dropped",
           any("/NOT_A_REGISTERED_COMMAND" in m.get_text() for m in combo.get_history_messages()))


    # ---------------------------------------------------------------- new topic
    # ---------------------------------------------------------------- developer prompt
    _section("6c. A developer-role prompt is treated exactly like a system prompt")

    #the prompt at message 0 may carry ROLE_DEVELOPER (add_message and load_conversation both put it
    #there). It used to be treated as an ordinary message by get_memory_messages() and fell out of
    #memory as soon as the window was full
    dev_conv = AIConversation(max_memory=2)
    dev_conv.add_message(rc.ROLE_DEVELOPER, "developer prompt")
    for i in range(4):
        dev_conv.add_message(rc.ROLE_USER, f"q{i}")
        dev_conv.add_message(rc.ROLE_ASSISTANT, f"a{i}")
    dev_memory = dev_conv.get_memory_messages()
    _check("add_message(ROLE_DEVELOPER) replaces message 0",
           dev_conv.get_ith_message(0).get_role() == rc.ROLE_DEVELOPER and dev_conv.count_all_messages() == 9)
    _check("and installs it sticky, like the system prompt", dev_conv.get_ith_message(0).is_sticky())
    _check("the developer prompt is in memory once the window is full",
           dev_memory[0].get_role() == rc.ROLE_DEVELOPER)
    _check("without using up a window slot", len(dev_memory) == 3, f"({len(dev_memory)})")
    _check("system token counter follows the developer prompt",
           dev_conv.get_system_tokens() == dev_conv.get_ith_message(0).get_estimated_tokens())
    sys_replaced = AIConversation()
    sys_replaced.add_message(rc.ROLE_SYSTEM, "replacement prompt")
    _check("add_message(ROLE_SYSTEM) also installs the prompt sticky", sys_replaced.get_ith_message(0).is_sticky())


    _section("7. new_topic() - a line under the conversation")

    before_topic = len(conv.get_memory_messages())
    conv.new_topic()
    conv.add_message(rc.ROLE_USER, "Totally new subject: what is 2+2?")
    after_topic = conv.get_memory_messages()

    print(f"  memory before /NEWTOPIC: {before_topic} messages")
    print(f"  memory after           : {len(after_topic)} messages -> {[m.get_text()[:32] for m in after_topic]}")

    _check("memory restarts at the marker", len(after_topic) < before_topic)
    _check("system prompt is still there", after_topic[0].get_role() == rc.ROLE_SYSTEM)
    _check("history keeps everything", conv.count_all_messages() > len(after_topic))
    #the "PIN THIS" message from section 6 is sticky but sits BEFORE the marker: memory never
    #reaches past the most recent /NEWTOPIC, not even for pinned messages
    _check("/NEWTOPIC beats sticky: a pinned message before the marker stays out",
           all("PIN THIS" not in m.get_text() for m in after_topic))


    # ---------------------------------------------------------------- token counters
    _section("8. Token counters (estimates, for display only)")

    print(f"  system prompt      : {conv.get_system_tokens()}")
    print(f"  user total         : {conv.get_user_tokens()}")
    print(f"  assistant total    : {conv.get_assistant_tokens()}")
    print(f"  whole conversation : {conv.get_total_tokens()}")
    print(f"  memory (exact)     : {conv.get_memory_total_tokens()}")
    print(f"  biggest user msg   : {conv.get_biggest_user_msg_tokens()}")
    print(f"  biggest reply      : {conv.get_biggest_assistant_msg_tokens()}")

    _check("total is positive", conv.get_total_tokens() > 0)
    _check("total covers user and assistant",
           conv.get_total_tokens() >= conv.get_user_tokens() + conv.get_assistant_tokens())
    _check("memory user tokens tracked", conv.get_memory_user_tokens() >= 0)
    _check("memory assistant tokens tracked", conv.get_memory_assistant_tokens() >= 0)

    edited = conv.get_ith_message(1)
    tokens_before = edited.get_estimated_tokens()
    edited.set_text("What is the capital city of the Italian Republic, and why?")
    conv.recalculate_tokens()
    _check("set_text refreshes the message estimate", edited.get_estimated_tokens() > tokens_before,
           f"({tokens_before} -> {edited.get_estimated_tokens()})")


    # ---------------------------------------------------------------- token accuracy
    _section("8b. Memory token counters match the real memory selection exactly")

    def _sum_tokens(messages) -> int:
        return sum(m.get_estimated_tokens() for m in messages)

    #a pinned message sitting outside the natural window must be COUNTED, not just displayed -
    #recalculate_tokens() used to derive memory totals from a positional cut over __messages that
    #knew nothing about stickiness, so this is exactly the case that cut got wrong
    tok_conv = AIConversation("System prompt for token accuracy checks.", max_memory=2)
    tok_conv.add_message(rc.ROLE_USER, "pinned question that must be counted", is_sticky=True)
    for i in range(4):
        tok_conv.add_message(rc.ROLE_USER, f"filler question number {i}")
        tok_conv.add_message(rc.ROLE_ASSISTANT, f"filler answer number {i}")

    expected_memory = tok_conv.get_memory_messages(add_sticky=True)
    expected_total  = _sum_tokens(expected_memory)
    expected_user   = _sum_tokens([m for m in expected_memory if m.get_role() == rc.ROLE_USER])
    expected_asst   = _sum_tokens([m for m in expected_memory if m.get_role() == rc.ROLE_ASSISTANT])

    print(f"  memory selection : {[m.get_text()[:28] for m in expected_memory]}")
    print(f"  reported total={tok_conv.get_memory_total_tokens()}, independently summed={expected_total}")

    _check("the pinned message outside the window is part of the real memory selection",
           any("pinned question" in m.get_text() for m in expected_memory))
    _check("memory_total_tokens matches an independent sum over that selection",
           tok_conv.get_memory_total_tokens() == expected_total)
    _check("memory_user_tokens matches the real selection",
           tok_conv.get_memory_user_tokens() == expected_user)
    _check("memory_assistant_tokens matches the real selection",
           tok_conv.get_memory_assistant_tokens() == expected_asst)

    #narrow the window: the pinned message stays in (it is sticky), totals must stay correct
    tok_conv.set_max_memory_messages(1)
    expected_memory2 = tok_conv.get_memory_messages(add_sticky=True)
    _check("memory totals stay correct after narrowing the window",
           tok_conv.get_memory_total_tokens() == _sum_tokens(expected_memory2))

    #cross a /NEWTOPIC marker: totals must reset to (system + post-marker) only, dropping
    #everything before it even though the pinned message is sticky
    tok_conv.new_topic()
    tok_conv.add_message(rc.ROLE_USER, "fresh topic question")
    expected_memory3 = tok_conv.get_memory_messages(add_sticky=True)
    print(f"  after /NEWTOPIC  : {[m.get_text()[:28] for m in expected_memory3]}, "
          f"reported={tok_conv.get_memory_total_tokens()}, expected={_sum_tokens(expected_memory3)}")

    _check("memory totals follow a /NEWTOPIC boundary correctly",
           tok_conv.get_memory_total_tokens() == _sum_tokens(expected_memory3))
    _check("the pinned pre-marker message no longer counts once a new topic starts",
           not any("pinned question" in m.get_text() for m in expected_memory3))

    #the "biggest" counters are recomputed from scratch every call, not tracked incrementally -
    #removing the biggest message must shrink them back down, not leave a stale high-water mark
    big_conv = AIConversation("System prompt.", max_memory=10)
    big_conv.add_message(rc.ROLE_USER, "short")
    big_conv.add_message(rc.ROLE_USER,
                          "a much longer user message with a lot more tokens in it than the others here")
    biggest_with = big_conv.get_biggest_user_msg_tokens()
    big_conv.remove_nth_message(big_conv.count_all_messages() - 1)          #drop the long one
    biggest_without = big_conv.get_biggest_user_msg_tokens()
    _check("the biggest-user-message counter shrinks once the biggest message is removed",
           biggest_without < biggest_with, f"({biggest_with} -> {biggest_without})")

    #num_tokens_from_string() must degrade gracefully, not raise, for a model tiktoken cannot place
    _fallback_msg = AIMessage(rc.ROLE_USER, "hello there", rc.AIC_TYPE_TEXT)
    _fallback_count = _fallback_msg.num_tokens_from_string("hello there", model_name="totally-unknown-model-xyz")
    _check("num_tokens_from_string falls back instead of raising for an unrecognised model name",
           _fallback_count > 0)


    # ---------------------------------------------------------------- inspection
    _section("9. Inspecting and counting")

    print(f"  all messages            : {conv.count_all_messages()}")
    print(f"  sticky / non sticky     : {conv.count_sticky_messages()} / {conv.count_non_sticky_messages()}")
    print(f"  count (history)         : {conv.get_messages_count()}")
    print(f"  count (memory only)     : {conv.get_messages_count(memory_only=True)}")
    print(f"  count (history+internal): {conv.get_messages_count(memory_only=False, count_internal=True)}")
    print(f"  memory size estimate    : {conv.get_memory_messages_no()}")
    print(f"  last message            : {conv.get_last_message_text()!r}")
    print(f"  message [2] text        : {conv.get_ith_message_text(2)!r}")

    _check("counting internals gives a bigger number",
           conv.get_messages_count(memory_only=False, count_internal=True) >
           conv.get_messages_count(memory_only=False, count_internal=False))
    #count_internal defaults to True: this count doubles as the index bound for /sticky and
    #/delete elsewhere in the app, and those indices are positions in the FULL message list
    _check("get_messages_count() defaults to counting internals (it is used as an index bound)",
           conv.get_messages_count() == conv.get_messages_count(count_internal=True))
    _check("get_messages_copy is a copy", conv.get_messages_copy() is not conv.get_messages())
    _check("out of range index raises", not _reads(conv, 9999))
    _check("negative index reads from the end",
           conv.get_ith_message_text(-1) == conv.get_last_message_text())

    printColor("\n  -- 1 print_message_No(3) --", "bold_yellow")
    conv.print_message_No(3)

    printColor("\n  -- 2 print_message_No(3) -- (text_only)", "bold_yellow")
    conv.print_message_No(3, True)

    printColor("  -- to_string(memory_only=True) --", "bold_yellow")
    print("   " + conv.to_string(memory_only=True).replace("\n", "\n   ")[:300] + " ...")

    record = conv.to_dict()
    _check("to_dict is a record of the conversation", set(record.keys()) == {"max_memory_messages", "messages"})
    _check("the record keeps internals", any(m["role"] == rc.ROLE_INTERNAL for m in record["messages"]))
    _check("the record keeps model provenance", any(m["model"] for m in record["messages"]))


    # ---------------------------------------------------------------- persistence
    _section("10. Saving and loading")

    temp_file:str = os.path.join(tempfile.gettempdir(), "aiconversation_selftest.jsonl")
    try:
        conv.save_conversation(temp_file)
        reloaded = AIConversation.from_file(temp_file)

        print(f"  saved to      : {temp_file}")
        print(f"  messages out  : {conv.count_all_messages()}, back in: {reloaded.count_all_messages()}")

        _check("message count survives the round trip",
               reloaded.count_all_messages() == conv.count_all_messages())
        _check("system prompt survives", reloaded.get_system_prompt() == conv.get_system_prompt())
        _check("text survives", reloaded.get_last_message_text() == conv.get_last_message_text())
        _check("image survives",
               any(m.get_type() == rc.AIC_TYPE_IMAGE_URL for m in reloaded.get_history_messages()))
        _check("internal commands survive",
               any(m.get_role() == rc.ROLE_INTERNAL for m in reloaded.get_history_messages()))
        _check("model provenance survives",
               any(m.get_model() for m in reloaded.get_history_messages()))
        _check("stickiness survives", reloaded.count_sticky_messages() == conv.count_sticky_messages())

        #from_file() builds a new object; load_conversation() refills an existing one
        refilled = AIConversation("this prompt is about to be replaced")
        refilled.load_conversation(temp_file)
        _check("load_conversation refills an existing conversation",
               refilled.count_all_messages() == conv.count_all_messages())
        _check("and replaces its system prompt", refilled.get_system_prompt() == conv.get_system_prompt())

        #the memory window is a generation setting, so it travels with the CALLER, not the file
        windowed = AIConversation.from_file(temp_file, max_memory=11)
        _check("from_file takes the memory window from the caller",
               windowed.get_max_memory_messages() == 11)
        _check("and defaults to the standard window when not told",
               AIConversation.from_file(temp_file).get_max_memory_messages() == rc.AIC_DEFAULT_MEMORY_SIZE)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("  temp file removed")


    # ---------------------------------------------------------------- damaged files
    _section("10b. A damaged file costs you the bad lines, not the conversation")

    broken_file:str = os.path.join(tempfile.gettempdir(), "aiconversation_broken.jsonl")
    try:
        with open(broken_file, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": rc.ROLE_SYSTEM, "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "recovered prompt", "sticky": True}) + "\n")
            handle.write("\n")                                    #blank line
            handle.write("{not json at all\n")                    #unparseable
            #a file written before 'sticky' and 'model' existed
            handle.write(json.dumps({"role": rc.ROLE_USER, "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "legacy question"}) + "\n")
            handle.write(json.dumps({"role": "wizard", "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "bad role", "sticky": False}) + "\n")
            handle.write(json.dumps({"role": rc.ROLE_ASSISTANT, "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "survivor", "sticky": False}) + "\n")
            handle.write("123' + '\n")                                  #valid JSON, but not an object
            #an image whose caption was saved as null: an empty caption, not an error
            handle.write(json.dumps({"role": rc.ROLE_USER, "msg_type": rc.AIC_TYPE_IMAGE_URL,
                                     "content_text": None, "content_image_url": DEMO_PNG}) + "\n")

        printColor("  (the skipped lines are reported below - that is the point)", "gray")
        damaged = AIConversation.from_file(broken_file)
        texts   = [m.get_text() for m in damaged.get_history_messages()]
        print(f"  salvaged: {texts}")

        _check("the good lines survive", "survivor" in texts)
        _check("a blank line is ignored", len(texts) == 4, f"({len(texts)})")
        _check("a JSON line that is not an object is skipped, not fatal", "survivor" in texts)
        _check("a null caption loads as an empty caption",
               any(m.get_type() == rc.AIC_TYPE_IMAGE_URL and m.get_text() == "" for m in damaged.get_history_messages()))
        _check("a legacy line without 'sticky' still loads", "legacy question" in texts)
        _check("the unparseable line is skipped", not any("not json" in t for t in texts))
        _check("the invalid role is skipped", "bad role" not in texts)
        _check("the system prompt is still message 0", damaged.get_ith_message(0).get_role() == rc.ROLE_SYSTEM)

        #a file with nothing salvageable must still leave a usable conversation
        with open(broken_file, "w", encoding="utf-8") as handle:
            handle.write("")
        empty_load = AIConversation.from_file(broken_file)
        _check("an empty file yields the default system prompt",
               empty_load.count_all_messages() == 1 and
               empty_load.get_system_prompt() == rc.AIC_DEFAULT_SYSTEM_PROMPT)
        empty_load.restart()                                       #used to raise IndexError
        _check("and the conversation is usable afterwards", empty_load.count_all_messages() == 1)

        #a prompt line that is not the first line: it goes to the front, it does not overwrite
        #whatever real message happened to be loaded first
        with open(broken_file, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": rc.ROLE_USER, "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "first line is a question"}) + "\n")
            handle.write(json.dumps({"role": rc.ROLE_SYSTEM, "msg_type": rc.AIC_TYPE_TEXT,
                                     "content_text": "late prompt"}) + "\n")
        late_prompt = AIConversation.from_file(broken_file)
        late_texts  = [m.get_text() for m in late_prompt.get_history_messages()]
        _check("a late system line becomes message 0", late_prompt.get_system_prompt() == "late prompt")
        _check("without losing the message that was there", "first line is a question" in late_texts)
    finally:
        if os.path.exists(broken_file):
            os.remove(broken_file)


    # ---------------------------------------------------------------- resizing and removing
    _section("11. Resizing the window, removing messages")

    #shown on a FRESH conversation: `conv` already carries a /NEWTOPIC marker from section 7, which
    #caps memory at everything after the marker no matter how wide the window is
    window_demo = AIConversation("You are a test assistant.", max_memory=8)
    for i in range(6):
        window_demo.add_message(rc.ROLE_USER, f"question {i}")
        window_demo.add_message(rc.ROLE_ASSISTANT, f"answer {i}")

    print(f"  12 messages + system prompt, window 8 -> memory {len(window_demo.get_memory_messages())}")
    window_demo.set_max_memory_messages(2)
    print(f"  window narrowed to 2                  -> memory {len(window_demo.get_memory_messages())}")
    _check("window resized", window_demo.get_max_memory_messages() == 2)
    _check("memory shrank", len(window_demo.get_memory_messages()) <= 3)

    window_demo.set_max_memory_messages(8)
    print(f"  window widened back to 8              -> memory {len(window_demo.get_memory_messages())}")
    _check("nothing was lost, memory grows back", len(window_demo.get_memory_messages()) > 3)
    _check("history was never affected", window_demo.count_all_messages() == 13)
    _check("max_memory must be positive", not _resizes(window_demo, 0))

    count_before = conv.count_all_messages()
    _check("remove_nth_message(0) refuses to touch the system prompt", not conv.remove_nth_message(0))
    _check("system prompt still present", conv.get_ith_message(0).get_role() == rc.ROLE_SYSTEM)
    _check("remove_nth_message(2) works", conv.remove_nth_message(2))
    _check("one message gone", conv.count_all_messages() == count_before - 1)

    removed = conv.remove_non_sticky_messages(2)
    print(f"  remove_non_sticky_messages(2) removed {removed}")
    _check("sticky messages were spared", conv.count_sticky_messages() > 0)

    removed = conv.remove_messages(2, remove_sticky=True)
    print(f"  remove_messages(2, remove_sticky=True) removed {removed}")
    _check("removing a negative number raises", not _removes(conv, -1))

    conv.restart()
    print(f"  after restart(): {conv.count_all_messages()} message(s)")
    _check("restart keeps only the system prompt", conv.count_all_messages() == 1)
    _check("and it is the system prompt", conv.get_ith_message(0).get_role() == rc.ROLE_SYSTEM)
    _check("token counters followed", conv.get_total_tokens() == conv.get_system_tokens())


    # ---------------------------------------------------------------- deprecated
    _section("12. Deprecated entry point")
    conv.print()


    # ---------------------------------------------------------------- verdict
    _section("Result")
    if _failures:
        printColor(f"  {len(_failures)} CHECK(S) FAILED:", "bold_red")
        for failure in _failures:
            printColor(f"    - {failure}", "light_red")
    else:
        printColor("  all checks passed", "bold_green")

    return not _failures


if __name__ == '__main__':
    raise SystemExit(0 if run_tests() else 1)
