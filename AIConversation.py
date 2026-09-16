# Author:  Antonio Romeo
# Date:    2026-07-21
# Description: Conversation model for talking to AI models through an OpenAI compatible API.
#
# Three classes:
#   AIMessageContent - the content of one message (text, image, file, internal command, reasoning)
#   AIMessage        - one message: role, content, stickiness, token estimate, model provenance
#   AIConversation   - the whole conversation: history, memory window, persistence, AI-facing arrays
#
# DESIGN RULE - this module knows about conversations, not about how a model must be driven.
# It holds no model name, no temperature, no max_tokens, and never decides what a model can accept.
# It produces the "messages" array; the calling program assembles the request:
#
#   body = {"model": m, "messages": conv.get_memory_messages_for_ai(), "temperature": 0.7}
#
# Anything the caller knows and this module does not (which model, what it supports) is passed IN,
# e.g. strip_types={AIC_TYPE_IMAGE_URL} to degrade images for a model without vision.
#
# MIT License (see the LICENSE file distributed with this project)

import base64
import json
import mimetypes
import os
import time
from typing import Any
import tiktoken
import rom_const as rc
from rom_printing_utils import printColor, printColorSameLine
#import rom_util as ru


class AIMessageContent:
    """ The content of a single message, in two representations at once:

        - the SEMANTIC fields (type, text, image_url, file_url), which are what gets persisted, and
        - the CONTENT BLOCKS (get_content()), the list of {"type": ...} entries an AI endpoint
          expects inside a message.

        The two really do differ: an image url is a flat string in the semantic fields, but a nested
        {"url": ...} object inside its content block. The blocks are rebuilt from the semantic
        fields every time set_content() runs, so saved conversations stay independent of any
        endpoint's request format: only this class knows the block layout.

        Supported types (rom_const): AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE,
        AIC_TYPE_INTERNAL and AIC_TYPE_REASONING.

        Images and files may be given as an http URL, as a "data:" URL, or as a local path: a local
        path is READ AND BASE64 ENCODED at construction, and the stored url becomes the data URL.
        The original path is not kept.
    """
    def __init__(self, msg_type: str, msg_image_url: "str | None" = "", msg_text: str = "", msg_file_url: "str | None" = "") -> None:
        self.__content:  list[dict[str, Any]] = []
        self.__type: str = msg_type
        self.__image_url: "str | None" = msg_image_url
        self.__file_url: "str | None" = msg_file_url
        self.__text: str = msg_text
        self.set_content(msg_type, msg_image_url, msg_text, msg_file_url)


    def to_dict(self)-> dict[str, Any]:
        """ The semantic fields as a plain dict. The content blocks are not included: they are derived
            data, rebuilt by set_content() whenever this content is reconstructed.
        """
        my_dict_version: dict[str, Any] = {
            'type': self.__type,
            'image_url': self.__image_url,
            'file_url': self.__file_url,
            'text': self.__text
        }
        return my_dict_version

    @classmethod
    def from_dict(cls, d):
        return cls(d['type'], d.get('image_url', ''), d.get('text', ''), d.get('file_url', ''))
    
    def get_type(self) -> str:
        return self.__type
    
    def get_image_url(self) -> "str | None":
        return self.__image_url

    def get_file_url(self) -> "str | None":
        return self.__file_url

    def get_text(self) -> str:
        return self.__text
    
    def get_content(self) -> list[dict[str, Any]]:
        return self.__content
    
    def set_content(self, msg_type: str, msg_image_url: "str | None" = "", msg_text: str = "", msg_file_url: "str | None" = "") -> None:
        """ Replace the content and rebuild the content blocks returned by get_content().

            For AIC_TYPE_IMAGE_URL / AIC_TYPE_FILE a value that is neither an http nor a "data:" URL
            is treated as a LOCAL PATH: the file is read, base64 encoded, and the stored url is
            replaced by the resulting data URL. A read failure is not fatal - it is printed and the
            url block carries an error string instead, so one unreadable attachment cannot destroy
            a whole conversation being loaded from disk.

            Raises ValueError for an unsupported msg_type.
        """
        self.__type: str = msg_type
        self.__image_url: "str | None" = msg_image_url
        self.__file_url: "str | None" = msg_file_url
        self.__text: str = msg_text

        if msg_type == rc.AIC_TYPE_IMAGE_URL:
            msg_image_url = msg_image_url or ""
            #any "data:" url is already encoded, whatever its mime: a local file whose type could not be
            #guessed as an image still has to survive set_text() and a reload without being re-read
            #from disk as if the data url were a path
            if msg_image_url.startswith("http") or msg_image_url.startswith("data:"):
                self.__content = [
                    {
                        "type": rc.AIC_TYPE_IMAGE_URL,
                        "image_url": {
                            "url": msg_image_url
                        }
                    },
                    {
                        "type": rc.AIC_TYPE_TEXT,
                        "text": msg_text
                    }
                ]
            else:
                try:
                    with open(msg_image_url, "rb") as image_file:
                        encoded_image: str = base64.b64encode(image_file.read()).decode("utf-8")
                    img_mime, _ = mimetypes.guess_type(msg_image_url)
                    if img_mime is None:
                        img_mime = "image/jpeg"
                    self.__image_url = f"data:{img_mime};base64,{encoded_image}"
                    self.__content = [
                        {
                            "type": rc.AIC_TYPE_IMAGE_URL,
                            "image_url": {
                                "url": f"data:{img_mime};base64,{encoded_image}"
                            }
                        },
                        {
                            "type": rc.AIC_TYPE_TEXT,
                            "text": msg_text
                        }
                    ]
                except OSError as e:
                    print("Error opening image file:", e)
                    self.__content = [
                        {
                            "type": rc.AIC_TYPE_IMAGE_URL,
                            "image_url": {
                                "url": f"Error opening image file: {msg_image_url}"
                            }
                        },
                        {
                            "type": rc.AIC_TYPE_TEXT,
                            "text": msg_text
                        }
                    ]

                
        elif msg_type == rc.AIC_TYPE_TEXT:
            self.__content = [
                {
                    "type": rc.AIC_TYPE_TEXT,
                    "text": msg_text
                }
            ]
        elif msg_type == rc.AIC_TYPE_FILE:
            msg_file_url = msg_file_url or ""
            if msg_file_url.startswith("http") or msg_file_url.startswith("data:"):
                self.__content = [
                    {
                        "type": rc.AIC_TYPE_FILE,
                        "file_url": {
                            "url": msg_file_url
                        }
                    },
                    {
                        "type": rc.AIC_TYPE_TEXT,
                        "text": msg_text
                    }
                ]
            else:
                try:
                    mime_type, _ = mimetypes.guess_type(msg_file_url)
                    if mime_type is None:
                        mime_type = "application/octet-stream"
                    with open(msg_file_url, "rb") as file_obj:
                        encoded_file: str = base64.b64encode(file_obj.read()).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{encoded_file}"
                    self.__file_url = data_url
                    self.__content = [
                        {
                            "type": rc.AIC_TYPE_FILE,
                            "file_url": {
                                "url": data_url
                            }
                        },
                        {
                            "type": rc.AIC_TYPE_TEXT,
                            "text": msg_text
                        }
                    ]
                except OSError as e:
                    print("Error opening file:", e)
                    self.__content = [
                        {
                            "type": rc.AIC_TYPE_FILE,
                            "file_url": {
                                "url": f"Error opening file: {msg_file_url}"
                            }
                        },
                        {
                            "type": rc.AIC_TYPE_TEXT,
                            "text": msg_text
                        }
                    ]
        elif msg_type == rc.AIC_TYPE_INTERNAL or msg_type == rc.AIC_TYPE_REASONING: #store a command, internal message... will be not sent to AI
            self.__content = [
                {
                    "type": msg_type,
                    "text": msg_text
                }
            ]            
        else:
            raise ValueError("AIMessageContent.set_content - Invalid message type: " + msg_type)
        
        
    def to_string(self) -> str:
        """ Human readable rendering of the content, for logs and debugging.
            A pure read: the content blocks are rebuilt by set_content() and cannot drift from the
            semantic fields, so there is nothing to re-sync here.
        """
        result:str= ""
        if self.__type == rc.AIC_TYPE_IMAGE_URL:
            result = f"Image URL: {self.__image_url}\nText: {self.__text}"
        elif self.__type == rc.AIC_TYPE_FILE:
            result = f"File  URL: {self.__file_url}\nText: {self.__text}"
        elif self.__type == rc.AIC_TYPE_TEXT:
            result = f"Text: {self.__text}"
        elif self.__type == rc.AIC_TYPE_INTERNAL:
            result = f"Text: {self.__text}"
        elif self.__type == rc.AIC_TYPE_REASONING:
            result = f"Text: {self.__text}"
        return result


class AIMessage:
    """ One message in a conversation: a role, its content, and the bookkeeping around it
        (stickiness, token estimate, and which model produced it).

        Roles and types are the rom_const values, all lowercase - ROLE_USER, ROLE_ASSISTANT,
        ROLE_SYSTEM, ROLE_DEVELOPER, ROLE_INTERNAL and AIC_TYPE_*.

        The `model` field is PROVENANCE only: it records which model produced (or received) this
        message so a reader can tell who said what. It is never sent as a request parameter - the
        calling program supplies the target model when it builds the request.
    """
    def __init__(self, role: str, content_text:str, msg_type:str, content_image_url:"str | None" = None, sticky:bool = False, content_file_url:"str | None" = None, model:str = "") -> None:
        """ Build a message. Every type validates its own required content.

        Args:
            role: one of ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_DEVELOPER, ROLE_INTERNAL.
            content_text: the text of the message. Required for every type except AIC_TYPE_IMAGE_URL
                and AIC_TYPE_FILE, where it acts as the caption and may be empty.
            msg_type: one of AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE, AIC_TYPE_INTERNAL,
                AIC_TYPE_REASONING - but only if the role is allowed to carry it, see below.
            content_image_url: required for AIC_TYPE_IMAGE_URL. http URL, "data:" URL or local path.
            sticky: sticky messages survive the memory window, see set_sticky().
            content_file_url: required for AIC_TYPE_FILE. Same accepted forms as the image url.
            model: provenance, the model that produced or received this message. Optional.

        The (role, type) contract is declared once in rc.AIC_ALLOWED_TYPES_BY_ROLE and enforced here:
        system/developer carry text only, user carries text/image/file, assistant carries those plus
        reasoning, and internal carries only its command text. An internal message therefore cannot
        hold an image, and reasoning is an assistant-only kind of output.

        Raises:
            ValueError: unknown role, unknown type, the content required by that type is missing, or a
                (role, type) pair the contract forbids (e.g. an image on an internal message).

        Notes:
            - Token estimates are computed here and refreshed by set_text(). AIC_TYPE_INTERNAL always
              costs 0 tokens because internal messages are never sent to a model.
            - set_text() changes the text in place; set_content() replaces the whole content and
              expects an AIMessageContent object.
        """
        self.__role: str = role
        self.__content: AIMessageContent
        self.__sticky: bool = sticky
        self.__model: str = model
        self.__estimated_tokens: int = 0
        self.__image_tokens: int = 0     #token cost of the image payload only. Fixed at construction, see __refresh_estimated_tokens()
        self.__is_internal: bool = False

        #an image or file may legitimately carry no caption: treat None as the empty string so the
        #token count and to_text_only_copy() never see a None (a saved file may hold null here)
        if content_text is None:
            content_text = ""

        #The whole (role, type) contract is one table, rc.AIC_ALLOWED_TYPES_BY_ROLE, checked here and
        #nowhere else. Three failures, most specific message for each:
        #  - unknown role: it has no row in the table (f-strings, not concatenation, so a corrupt
        #    saved message carrying role=None is REPORTED, not a TypeError inside the error path);
        #  - unknown type: not one of the known content types at all;
        #  - illegal pair: a known role carrying a type its row does not permit - e.g. an internal
        #    (command) message may hold only its command text, never an image, and reasoning is an
        #    assistant-only kind of output. This is enforced on construction AND on load (from_dict
        #    runs through here), so a saved file with an illegal pair is refused rather than trusted.
        allowed_types = rc.AIC_ALLOWED_TYPES_BY_ROLE.get(role)
        if allowed_types is None:
            raise ValueError(f"AIMessage.__init__ - Invalid message role: {role!r}")

        if (msg_type not in [rc.AIC_TYPE_TEXT, rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE, rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING]):
            raise ValueError(f"AIMessage.__init__ - Invalid message type: {msg_type!r}")

        if (msg_type not in allowed_types):
            raise ValueError(f"AIMessage.__init__ - role {role!r} may not carry {msg_type!r} content; "
                             f"allowed for this role: {sorted(allowed_types)}")

        if (msg_type == rc.AIC_TYPE_IMAGE_URL):
            if content_image_url is None or content_image_url == "":
                raise ValueError("Missing Image URL for " + msg_type + " message")
            else:
                self.__content = AIMessageContent(rc.AIC_TYPE_IMAGE_URL, content_image_url, content_text)
                self.__image_tokens = self.num_tokens_from_picture(content_image_url)
        elif (msg_type == rc.AIC_TYPE_INTERNAL):
            if content_text is None or content_text == "":
                raise ValueError("Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(msg_type, None, content_text)
            self.__is_internal = True
        elif (msg_type == rc.AIC_TYPE_REASONING):
            if content_text is None or content_text == "":
                raise ValueError("Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_REASONING, None, content_text)
            self.__is_internal = True
        elif (msg_type == rc.AIC_TYPE_TEXT):
            if content_text is None or content_text == "":
                raise ValueError("AIMessage.__init__ - Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_TEXT, None, content_text)
        elif (msg_type == rc.AIC_TYPE_FILE):
            if content_file_url is None or content_file_url == "":
                raise ValueError("Missing File URL for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_FILE, msg_file_url=content_file_url, msg_text=content_text)
        else:
            raise ValueError("AIMessage.__init__ - Invalid message type: " + msg_type)

        self.__refresh_estimated_tokens()

    def __refresh_estimated_tokens(self) -> None:
        """ Recompute the token estimate from the CURRENT content. Single source of truth for the
            per-type rules, used both at construction and whenever the text is replaced.

            - INTERNAL and REASONING messages are never sent to AI as part of memory, so they always
              cost 0. Reasoning is model output the user paid to generate, but it is excluded from
              the memory the model sees (see __may_enter_memory), so counting its tokens would only
              inflate the memory estimate with text that is never sent. Treat it like INTERNAL.
            - Every other type costs the tokens of its text, plus (for images) the cached image cost.

            The image cost is deliberately NOT recomputed here: it is derived once in __init__ from the
            URL as it was originally passed in. Recomputing it from the stored URL would change the
            estimate of an image message just because its text was edited (a local file path gets
            normalised into a much longer base64 data URL by AIMessageContent).
        """
        if self.__content.get_type() in (rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING):
            self.__estimated_tokens = 0     #not sent to AI as part of memory, so no token no cry
        else:
            self.__estimated_tokens = self.__image_tokens + self.num_tokens_from_string(self.__content.get_text())
        return

    def num_tokens_from_picture(self, encoded_image:str) -> int:
        """ Rough token estimate for an image, derived from the LENGTH of the encoded string.

            KNOWN LIMITATION - this is an approximation and can be far off. Real vision pricing is
            driven by PIXEL DIMENSIONS (a base cost plus a per-tile cost), not by byte size, and the
            two barely correlate: a heavily compressed 4000x3000 photo is small in bytes but
            expensive in tokens, while a small uncompressed bitmap is the opposite.

            It is also passed the url AS ORIGINALLY GIVEN, so an image supplied as a local path is
            measured on the length of the path string (~1024 tokens whatever the real image is),
            while the same image supplied as a data URL measures realistically.

            Good enough for the informational counters on screen; do not budget against it. The
            authoritative number is the usage returned by the API in its response.
        """
        num_tokens: int = 8192  #for a 1024x1024 image
        #let's write an implementation based on the size of the base64 string
        if encoded_image:
            base64_size:int = len(encoded_image)
            #each 3 bytes of data are represented by 4 bytes of base64
            image_size_bytes:int = (base64_size * 3) // 4
            #for DALL-E 3 each 256x256 image is 1024 tokens
            num_256_blocks:int = (image_size_bytes + (256*256 -1)) // (256*256)
            num_tokens = num_256_blocks * 1024
        
        return num_tokens

    def num_tokens_from_string(self, the_string: str, model_name: str = rc.AIC_MODEL_NAME_FOR_TOKEN_COUNT) -> int:
        """ Token count for a string, using the tiktoken encoding of `model_name` (e.g. gpt-4o).

            tiktoken maps unknown "gpt-*" names onto a default encoding, but raises KeyError for a
            name it cannot place at all (e.g. "claude-3", "mistral-large"). Since that would break
            the construction of EVERY message, an unknown name falls back to the o200k_base encoding
            instead - the count is then an approximation for that provider, which is all the on
            screen counters need.
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            #model unknown to tiktoken: fall back to the encoding of the representative model
            encoding = tiktoken.get_encoding("o200k_base")
        num_tokens: int = len(encoding.encode(the_string))
        return num_tokens

    def get_estimated_tokens(self) -> int:
        return self.__estimated_tokens
    
    def to_dict(self) -> dict[str, Any]:
        """ Serialise to the plain dict used on disk (one JSON line per message).

            Semantic fields only - the content blocks are NOT stored, they are rebuilt on load. That is
            what keeps saved conversations independent of any endpoint's request format.
            Note that image and file urls are stored in their normalised "data:" form.
        """
        my_dict_version: dict[str, Any] = {
            'role': self.__role,
            'msg_type': self.__content.get_type(),
            'content_text': self.__content.get_text(),
            'content_image_url': self.__content.get_image_url(),
            'content_file_url': self.__content.get_file_url(),
            'sticky': self.__sticky,
            'model': self.__model
        }
        return my_dict_version
    
    def to_string(self) -> str:
        """ Human readable rendering of role, content and stickiness, for logs and debugging.
            For terminal display prefer AIConversation.print() and its show_* wrappers.
        """
        result:str= f"Role: {self.__role}\nContent: {self.__content.to_string()}\nSticky: {self.__sticky}"
        return result


    @classmethod
    def from_dict(cls, d):
        """ Rebuild a message from the dict written by to_dict().
            Every optional key is read with a default, so a file written by an older version (one
            without `model`, say) still loads.
        """
        return cls(
            role = d.get('role'),
            content_text = d.get('content_text', ""),
            msg_type = d.get('msg_type', rc.AIC_TYPE_TEXT),
            content_image_url = d.get('content_image_url', None),
            sticky = bool(d.get('sticky', False)),
            content_file_url = d.get('content_file_url', None),
            model = d.get('model', '')
        )
        
    def get_type(self) -> str:
        return self.__content.get_type()
    
    def get_url(self) -> "str | None":
        """ The IMAGE url (normalised to a data URL when it came from a local file).
            For file attachments use get_file_url() instead.
        """
        return self.__content.get_image_url()

    def get_file_url(self) -> "str | None":
        return self.__content.get_file_url()
    
    def get_text(self) -> str:
        return self.__content.get_text()
    
    def set_text(self, text: str) -> None:
        """ Replace the text of the message and refresh its token estimate.
            NOTE: callers holding this message inside an AIConversation should call
            AIConversation.recalculate_tokens() afterwards so the conversation totals follow.
        """
        self.__content.set_content(self.__content.get_type(), self.__content.get_image_url(), text, self.__content.get_file_url())
        self.__refresh_estimated_tokens()
        return
    
    def get_role(self) -> str:
        return self.__role

    def set_role(self, role: str) -> None:
        """ Change the role of the message, subject to the same (role, type) contract the constructor
            enforces: the new role must exist and must be allowed to carry this message's content type.
            So an image message cannot become a system or internal message, and an internal command
            cannot be re-roled as conversation.

            Raises ValueError, leaving the message untouched, when the pair is not permitted.

            NOTE: callers holding this message inside an AIConversation should call
            AIConversation.recalculate_tokens() afterwards, as the user/assistant token split follows
            the role and the conversation cannot observe this change.
        """
        allowed_types = rc.AIC_ALLOWED_TYPES_BY_ROLE.get(role)
        if allowed_types is None:
            raise ValueError(f"AIMessage.set_role - Invalid message role: {role!r}")

        msg_type: str = self.__content.get_type()
        if msg_type not in allowed_types:
            raise ValueError(f"AIMessage.set_role - role {role!r} may not carry {msg_type!r} content; "
                             f"allowed for this role: {sorted(allowed_types)}")

        self.__role = role

    def get_model(self) -> str:
        return self.__model

    def set_model(self, model: str) -> None:
        self.__model = model

    def set_sticky(self, sticky: bool) -> None:
        """ Sticky messages are not removed from the conversation memory when a new message is added. 
            They are kept in the conversation until the user decides to remove them. 
            However there is a flag in the removemessages method to remove sticky messages as well.
        """
        self.__sticky = sticky

    def is_sticky(self) -> bool:
        return self.__sticky

    def is_internal(self) -> bool:
        """ True for AIC_TYPE_INTERNAL and AIC_TYPE_REASONING content.

            CAREFUL: this is a property of the TYPE, and the two types behave differently.
            AIC_TYPE_INTERNAL always implies ROLE_INTERNAL (the constructor refuses any other
            pairing) and never reaches memory. AIC_TYPE_REASONING normally carries role assistant,
            reports True here, and IS part of memory when add_reasoning=True.
        """
        return self.__is_internal

    def set_content(self, content: AIMessageContent) -> None:
        """ Replace the whole content object.
            NOTE: unlike set_text() this does NOT refresh the token estimate - the caller owns the
            new content and must know what it costs.
        """
        self.__content: AIMessageContent = content

    def get_content(self) -> AIMessageContent:
        return self.__content

    def to_text_only_copy(self, placeholder: str = rc.AIC_PLACEHOLDER_GENERIC) -> "AIMessage":
        """ Return a NEW text-only message standing in for this one, with the attachment replaced by
            `placeholder`. Role, stickiness and model are preserved; the original is left untouched.

            The message itself is kept rather than dropped on purpose: an image or file message
            usually carries a caption, and the reply that follows refers to it. Removing the whole
            turn would orphan that reply and break user/assistant alternation.
        """
        original_text: str = self.get_text().strip()
        new_text: str = f"{original_text}\n{placeholder}" if original_text else placeholder

        return AIMessage(self.__role, new_text, rc.AIC_TYPE_TEXT, None, self.__sticky, None, self.__model)

    def get_message_payload(self) -> dict:
        """ This message as one entry of an AI "messages" array: {"role": ..., "content": [blocks]}.

            Only role and content cross the boundary. Everything else this class tracks - stickiness,
            token estimate, model provenance - is bookkeeping and stays behind.

            Normally reached through AIConversation.get_memory_messages_for_ai().
        """
        payload = {
            "role": self.get_role(),
            #"sticky": self.is_sticky(),
            "content": self.get_content().get_content()
        }
        return payload


class AIConversation:
    """ A conversation between a user and an AI assistant.

        TWO VIEWS OF THE SAME MESSAGES
        ------------------------------
        HISTORY is everything that ever happened, in order, including the internal command markers.
        It is the record: get_history_messages(), print(), save_conversation().

        MEMORY is the subset a model is allowed to see: the last `max_memory` messages, plus the
        system prompt, plus any sticky messages that fell outside the window, and nothing from
        before the most recent /NEWTOPIC marker. Internal messages are ALWAYS excluded, and
        reasoning messages are excluded unless add_reasoning=True.

        Message 0 is always the system prompt and is never removed by restart() or remove_*().

        FOUR ACCESSORS, TWO AXES
        ------------------------
                        AIMessage objects            plain dicts for an AI endpoint
            memory      get_memory_messages()        get_memory_messages_for_ai()
            history     get_history_messages()       get_history_messages_for_ai()

        The *_for_ai() pair returns ONLY the messages array. Model, temperature and max_tokens are
        the calling program's business, see the module header.

        Args:
            system_message: the system prompt. Falls back to AIC_DEFAULT_SYSTEM_PROMPT when empty.
            max_memory: how many messages (excluding the system prompt) the memory window holds.
    """
    def __init__(self, system_message:"str | None" = None,  max_memory:int = rc.AIC_DEFAULT_MEMORY_SIZE) -> None:
        self.__messages: list[AIMessage]

        self.__max_memory_messages: int     = max_memory #the maximum number of messages to be used in the conversation with AI (NOT including the system message)

        self.__system_message_tokens: int   = 0          #tokens used by the system message

        self.__user_tokens: int             = 0          #tokens used by the user messages (whole history)
        self.__assistant_tokens: int        = 0          #tokens used by the assistant messages (whole history)
        self.__total_tokens: int            = 0          #the total number of tokens in the conversation. calculated automatically at every insert/remove
        self.__biggest_user_msg_tokens: int         = 0          
        self.__biggest_assistant_msg_tokens: int    = 0          
        
        self.__memory_user_tokens: int      = 0          #the total number of tokens in the conversation memory (i.e. limited to the last __maxmero). calculated automatically at every insert/remove
        self.__memory_assistant_tokens: int = 0          #the total number of tokens in the conversation. calculated automatically at every insert/remove
        self.__memory_total_tokens: int     = 0          #the total number of tokens in the conversation. calculated automatically at every insert/remove

        the_system_msg:str = ""

        #enforce rules for messages in the conversation:
        #1 there can be only 1 system message at the beginning of the conversation
        if system_message is not None and len(system_message.strip()) > 0:
            the_system_msg = system_message.strip()
        else:
            the_system_msg = rc.AIC_DEFAULT_SYSTEM_PROMPT
            #create a standard system message
        
        self.__messages = [AIMessage(rc.ROLE_SYSTEM, the_system_msg, rc.AIC_TYPE_TEXT, None, True)]
        
        self.__total_tokens = self.__messages[0].get_estimated_tokens()
        self.__system_message_tokens  = self.__total_tokens
        self.__memory_total_tokens    = self.__total_tokens

        self.__max_memory_messages = max_memory
        return


     
    @classmethod
    def from_file(cls, filename, max_memory:int = rc.AIC_DEFAULT_MEMORY_SIZE):
        """ Build a conversation from a file previously written by save_conversation().

            The system prompt comes from the file. The MEMORY WINDOW does not: it is a generation
            setting, not part of the record of what was said, so the caller supplies it exactly as it
            supplies the model and the temperature (see the module header). Pass the same max_memory
            you would pass to the constructor.
        """
        # Create an instance of AIConversation starting from a file
        conversation = cls(max_memory=max_memory)

        # Load the conversation from the file
        conversation.load_conversation(filename)

        return conversation

    def get_biggest_user_msg_tokens(self) -> int:
        return self.__biggest_user_msg_tokens
    
    def get_biggest_assistant_msg_tokens(self) -> int:
        return self.__biggest_assistant_msg_tokens

    def get_system_tokens(self) -> int:
        return self.__system_message_tokens

    def get_memory_total_tokens(self) -> int:
        """ Estimated tokens in the memory window.
            APPROXIMATE - see the note in recalculate_tokens() about how the window is measured here.
        """
        return self.__memory_total_tokens

    def get_total_tokens(self) -> int:
        return self.__total_tokens

    def get_memory_user_tokens(self) -> int:
        return self.__memory_user_tokens
    
    def get_memory_assistant_tokens(self) -> int:
        return self.__memory_assistant_tokens
    
    def get_user_tokens(self) -> int:
        return self.__user_tokens
    
    def get_assistant_tokens(self) -> int:
        return self.__assistant_tokens


    def load_conversation(self, full_path_name:str) -> None:
        """ Replace this conversation with the one stored in `full_path_name`.

            Format is JSON Lines: one message per line, in the shape AIMessage.to_dict() writes.
            Only semantic fields are stored, so the content blocks are rebuilt on load and a file saved
            by an older version stays readable - every optional key is read through AIMessage.from_dict(),
            which defaults anything missing.

            A line that fails is REPORTED AND SKIPPED rather than aborting the load: one corrupt
            message should not cost the user the whole conversation. That covers unparseable JSON,
            missing required fields and content a message type rejects. Blank lines are ignored.
            Only a failure to READ the file propagates.

            The memory window is NOT read from the file - it belongs to the caller, see from_file().

            Nothing is destroyed until the file has been read: the messages are built into a separate
            list and only swapped in at the end, so a file that cannot be opened or read leaves this
            conversation exactly as it was.

            Once the read succeeds the replacement does happen, even if every line was skipped - but
            the result is a conversation with the DEFAULT system prompt, never an empty object with
            no message 0. Callers that must not lose the current conversation on a damaged file
            should load into a new instance with from_file() and swap it in themselves.
        """
        loaded: list[AIMessage] = []
        line_number:int = 0

        with open(full_path_name, "r", encoding="utf-8") as file:
            for line in file:
                line_number += 1
                stripped_line:str = line.strip()
                if not stripped_line:
                    continue

                try:
                    message_dict = json.loads(stripped_line)
                    if not isinstance(message_dict, dict):
                        raise TypeError(f"expected a JSON object, got {type(message_dict).__name__}")
                    new_message  = AIMessage.from_dict(message_dict)
                except (ValueError, KeyError, TypeError) as e:
                    #ValueError covers json.JSONDecodeError and every AIMessage validation failure
                    print(f"AIConversation.load_conversation() - Error loading line {line_number}: {e}")
                    print(f"Wrong Line {line_number}: {stripped_line[:200]}")
                    continue

                #same rule as add_message: one system prompt, always at the front. It REPLACES a
                #prompt already there, but never a real message that happens to sit at index 0
                if new_message.get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                    if loaded and loaded[0].get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                        loaded[0] = new_message
                    else:
                        loaded.insert(0, new_message)
                else:
                    loaded.append(new_message)

        #message 0 is always the system prompt: the whole class relies on it, so a file that is empty,
        #truncated, or simply does not start with one gets the default rather than leaving the
        #conversation in a state where restart() and get_last_message_text() cannot work
        if not loaded or loaded[0].get_role() not in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
            printColor(f"AIConversation.load_conversation() - no system message found in {full_path_name}, using the default prompt.", "yellow")
            loaded.insert(0, AIMessage(rc.ROLE_SYSTEM, rc.AIC_DEFAULT_SYSTEM_PROMPT, rc.AIC_TYPE_TEXT, None, True))

        self.__restart_internal()
        self.__messages = loaded
        self.recalculate_tokens()
        return

    def save_conversation(self, full_path_name:str, max_retries:int = 3) -> None:
        """ Write the whole conversation (history, internal messages included) as JSON Lines.

            Written safely: the data goes to a ".tmp" file which is flushed and fsync'd, then
            atomically renamed over the target, so an interrupted save cannot leave a half written
            conversation behind. Transient I/O errors are retried with exponential backoff, which
            matters on synced folders (OneDrive, network shares) that briefly lock files.

            Only the messages are written. The memory window is a generation setting owned by the
            caller, not part of the record - hand it back through from_file(path, max_memory=...).

            Args:
                full_path_name: destination. Its directory must already exist.
                max_retries: attempts before giving up.

            Raises:
                RuntimeError: every attempt failed. The original error is chained as __cause__.
        """
        temp_path = full_path_name + '.tmp'
        last_error = None

        for attempt in range(max_retries):
            try:
                with open(temp_path, "w", encoding="utf-8") as file:
                    for message in self.__messages:
                        file.write(json.dumps(message.to_dict()) + '\n')
                    file.flush()
                    os.fsync(file.fileno())

                # Atomic rename: replace original with temp file
                if os.path.exists(full_path_name):
                    os.replace(temp_path, full_path_name)
                else:
                    os.rename(temp_path, full_path_name)
                # Some network/sync filesystems (e.g. SMB shares of a remote
                # OneDrive folder) carry over a stale mtime through MoveFileEx,
                # so the file listing's "Last Modified" never refreshes.
                # Force-stamp the current time.
                try:
                    os.utime(full_path_name, None)
                except OSError:
                    pass
                return

            except (OSError, IOError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 1 * (2 ** attempt)
                    print(f"I/O error saving conversation (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed to save conversation after {max_retries} attempts: {full_path_name}") from last_error
            finally:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    def change_system_message(self, new_system_message: str) -> None:
        """ Replace the system prompt at position 0, keeping the rest of the conversation.
            The replacement is sticky, like the original. Token counters are recalculated.
        """
        if len(self.__messages) == 0:
            self.__messages.append(AIMessage(rc.ROLE_SYSTEM, new_system_message, rc.AIC_TYPE_TEXT, None, True))
        else:
            self.__messages[0] = AIMessage(rc.ROLE_SYSTEM, new_system_message, rc.AIC_TYPE_TEXT, None, True)
        self.recalculate_tokens()
        return

    def __restart_internal(self) -> None:
        """ Private method to allow removing all messages from the conversation INCLUDING the system message."""
        self.__messages = []
        self.__system_message_tokens = 0
        self.__user_tokens = 0
        self.__assistant_tokens = 0
        self.__total_tokens = 0
        self.__memory_user_tokens = 0
        self.__memory_assistant_tokens = 0
        self.__memory_total_tokens = 0
        self.__biggest_user_msg_tokens = 0          
        self.__biggest_assistant_msg_tokens = 0          

        return

    def new_topic(self) -> None:
        """ Draw a line under the conversation so far: append an internal AIC_COMMAND_NEWTOPIC marker.

            From here on memory stops at this marker, so the model sees the system prompt plus only
            what follows it. Nothing is deleted: the earlier messages stay in history and are still
            saved, shown by print() and returned by get_history_messages().
        """
        self.add_message(rc.ROLE_INTERNAL, rc.AIC_COMMAND_NEWTOPIC, rc.AIC_TYPE_INTERNAL)
        return
    
    def add_comment(self, text:str) -> None:
        """ Record a note in the conversation: something that HAPPENED rather than something that was
            said. Stored as ROLE_INTERNAL + AIC_TYPE_INTERNAL, so it appears in the history and in the
            saved file, is never sent to a model, and costs no tokens.

            The application decides what is worth recording - a parameter change ("/MODEL:gpt-4o"), or
            any free text. The text is stored VERBATIM: this class does not parse it, whitelist it or
            change its case.

            (This used to only store strings beginning with one of six known prefixes and silently
            discard everything else, so forgetting to register a new command in a list on the other
            side of the codebase made it vanish without a word.)
        """
        self.add_message(rc.ROLE_INTERNAL, text, rc.AIC_TYPE_INTERNAL)
        return


    def add_message(self, msg_role:str, msg_text:str, msg_type:str=rc.AIC_TYPE_TEXT, image_url:"str | None" = None, is_sticky = False, file_url:"str | None" = None, model:str = "") -> None:
        """ Append a message to the conversation.

            User, assistant and internal messages are appended, as many as you like. A SYSTEM or
            DEVELOPER message is not appended: it REPLACES message 0, so a conversation always has
            exactly one system prompt and it always sits at the front.

            Token counters are recalculated on every call.

            Args:
                msg_role: ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_DEVELOPER or ROLE_INTERNAL.
                msg_text: the text of the message (the caption, for an image or file message).
                msg_type: one of the AIC_TYPE_* values. Defaults to AIC_TYPE_TEXT.
                image_url: required when msg_type is AIC_TYPE_IMAGE_URL.
                is_sticky: keep this message in memory even once it falls outside the window.
                file_url: required when msg_type is AIC_TYPE_FILE.
                model: provenance, which model produced or received this message.

            Raises:
                ValueError: unknown role or type, content missing for the given type, or
                    AIC_TYPE_INTERNAL used with a role other than ROLE_INTERNAL (see AIMessage).
        """
        #the prompt at message 0 is always sticky, like the one __init__ and change_system_message() install
        is_still_sticky:bool = is_sticky or msg_role in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]
        if msg_role not in [rc.ROLE_USER, rc.ROLE_ASSISTANT, rc.ROLE_SYSTEM, rc.ROLE_INTERNAL, rc.ROLE_DEVELOPER]:
            raise ValueError(f"AIConversation.add_message - Invalid message role: {msg_role!r}")

        if msg_type not in [rc.AIC_TYPE_TEXT, rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE, rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING]:
            raise ValueError(f"AIConversation.add_message - Invalid message type: {msg_type!r}")

        new_msg = AIMessage(msg_role, msg_text, msg_type, image_url, is_still_sticky, file_url, model)

        if msg_role in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
            if len(self.__messages) >= 1:
                self.__messages[0] = new_msg
            else:
                self.__messages.append(new_msg)
        elif msg_role in [rc.ROLE_USER, rc.ROLE_ASSISTANT, rc.ROLE_INTERNAL]:
            self.__messages.append(new_msg)

        self.recalculate_tokens()
        return
    
    @staticmethod
    def __may_enter_memory(message:AIMessage, add_reasoning:bool) -> bool:
        """ Can this message be part of memory at all?

            Internal messages never can: they are commands, not conversation. Both the role and the
            type are checked - the constructor now refuses to pair AIC_TYPE_INTERNAL with any other
            role, but a conversation saved before that rule existed can still hold such a message.
            Reasoning messages only when explicitly asked for: they record HOW an answer was
            reached, which is worth keeping in the history but is not part of the dialogue.
        """
        result = True
        if message.get_role() == rc.ROLE_INTERNAL or message.get_type() == rc.AIC_TYPE_INTERNAL:
            result = False

        if message.get_type() == rc.AIC_TYPE_REASONING and not add_reasoning:
            result = False

        return result


    def get_memory_messages(self, add_sticky:bool = True, add_reasoning:bool = False, strip_types:"set[str] | None" = None) -> list[AIMessage]:
        """
        Retrieve the list of messages that are part of the conversation memory (the AI-facing subset).

        This method returns a subset of the conversation messages that fit within the memory limit (`__max_memory_messages`).
        It includes sticky messages and handles special cases, such as detecting a new topic.

        Internal messages (ROLE_INTERNAL: commands, topic markers) are ALWAYS excluded from memory, since
        they are never sent to the AI model. There is therefore no `ignore_internals` option here -- use
        get_history_messages() if you need a view that can include internal messages.

        Reasoning messages (AIC_TYPE_REASONING) are excluded too, unless add_reasoning is True. They
        record how an answer was reached rather than what was said, so they belong to the history.

        Neither kind consumes a slot in the window when excluded: hiding them makes room for more
        real conversation rather than wasting the budget on messages nobody will see.

        The function works as follows:
        - Iterates through the messages in reverse order (starting from the most recent).
        - Adds messages to the memory list until the memory limit is reached.
        - Includes sticky messages even if they exceed the memory limit.
        - Stops adding messages if a "new topic" internal message (`AIC_COMMAND_NEWTOPIC`) is encountered.
        - Ensures the system message (first message in the conversation) is included if a new topic is found.

        Args:
            add_sticky (bool): include sticky messages that fall outside the memory window. This
                EXPANDS the selection - it reaches outside the window and pulls messages back in.
            add_reasoning (bool): keep AIC_TYPE_REASONING messages. Unlike add_sticky this never
                expands the selection: it only stops excluding reasoning that is already inside the
                window. A sticky reasoning message outside the window is still left out.
            strip_types (set[str] | None): content types the CALLER knows the target model cannot
                accept, e.g. {AIC_TYPE_IMAGE_URL}. Matching messages are replaced by a text-only
                stand-in carrying their original text plus a placeholder. Deciding what a model can
                accept is the caller's job: this class holds no model knowledge and never guesses.

        Returns:
            list[AIMessage]: A list of `AIMessage` objects representing the conversation memory.

        Notes:
            - Sticky messages are always included in the memory, even if they exceed the memory limit.
            - If a "new topic" internal message is found, the system message is added to the memory.
            - Stripping NEVER alters stored history: substitutes are fresh objects, so the same
              conversation can be sent to a model with vision and one without, in any order.

        Example:
            memory_messages = conversation.get_memory_messages()
            for message in memory_messages:
                print(message.get_text())

            #target model is text only
            safe = conversation.get_memory_messages(strip_types={rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE})

            #feed the model back its own reasoning
            with_thinking = conversation.get_memory_messages(add_reasoning=True)
        """
        temp_memory_list: list[AIMessage] = []
        new_topic_found: bool = False
        system_message_found = False

        for message in reversed(self.__messages):

            if message.get_role() == rc.ROLE_INTERNAL and message.get_text() == rc.AIC_COMMAND_NEWTOPIC:
                new_topic_found = True
                break
            elif message.get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                #the prompt, whichever role it carries: always in memory, never a window slot
                system_message_found = True
                new_topic_found      = True
                temp_memory_list.append(message)
                break

            if len(temp_memory_list) < self.__max_memory_messages:
                #inside the window. Messages that may not enter memory are skipped WITHOUT using up
                #a window slot, so hiding them buys room for real conversation instead of wasting it
                if self.__may_enter_memory(message, add_reasoning):
                    temp_memory_list.append(message)
                continue
            else:
                #outside the window only STICKY messages are pulled back in, and never reasoning:
                #add_reasoning keeps reasoning that is already inside the window, it does not widen it
                if message.is_sticky() and add_sticky and self.__may_enter_memory(message, False):
                    temp_memory_list.append(message)
                continue
            
        if new_topic_found and not system_message_found:          #found a new topic (internal) message so we need to add system message
            temp_memory_list.append(self.__messages[0])

        temp_memory_list.reverse()

        if strip_types:
            temp_memory_list = self.strip_message_types(temp_memory_list, strip_types)

        return temp_memory_list


    @staticmethod
    def strip_message_types(messages:list[AIMessage], strip_types:"set[str]") -> list[AIMessage]:
        """ Return a copy of `messages` in which every message whose type is in `strip_types` is
            replaced by a text-only stand-in. Messages of other types are passed through unchanged
            (same objects), so only the substitutes are new.

            Exposed as a static helper so a caller can apply the same stripping to a history payload
            or to a list it assembled itself.
        """
        stripped_list: list[AIMessage] = []

        for message in messages:
            if message.get_type() in strip_types:
                placeholder: str = rc.AIC_PLACEHOLDER_BY_TYPE.get(message.get_type(), rc.AIC_PLACEHOLDER_GENERIC)
                stripped_list.append(message.to_text_only_copy(placeholder))
            else:
                stripped_list.append(message)

        return stripped_list


    def count_memory_types(self, add_sticky:bool = True, add_reasoning:bool = False) -> dict[str, int]:
        """ Count the memory messages by content type, e.g. {"text": 4, "image_url": 2}.
            Lets a caller report "2 images omitted" before deciding what to strip.
            Counts what memory would actually contain, so reasoning is absent unless asked for.
        """
        counts: dict[str, int] = {}

        for message in self.get_memory_messages(add_sticky=add_sticky, add_reasoning=add_reasoning):
            counts[message.get_type()] = counts.get(message.get_type(), 0) + 1

        return counts


    def get_history_messages(self, ignore_internals:bool = False) -> list[AIMessage]:
        """
        Retrieve the FULL conversation history (no memory limit, no sticky/new-topic logic).

        This is the human-facing counterpart of get_memory_messages(): it returns every message ever
        added to the conversation, in insertion order, starting with the system message at index 0.

        Args:
            ignore_internals (bool): If True, ROLE_INTERNAL messages (commands such as /MODEL,
                /MAXMESSAGES and the /NEWTOPIC marker) are filtered out. Default is False, i.e. the
                history shows everything -- internal messages are part of the record of what happened.

        Returns:
            list[AIMessage]: The conversation history. When ignore_internals is False the internal
            list is returned as-is (do not mutate it); when True a new filtered list is returned.

        Note:
            This list is NEVER used to build an AI payload -- see get_memory_messages() for that.
        """
        if not ignore_internals:
            return self.__messages

        return [message for message in self.__messages if message.get_role() != rc.ROLE_INTERNAL]


    @staticmethod
    def __to_ai_message_array(messages:list[AIMessage]) -> list[dict[str, Any]]:
        """ Turn AIMessage objects into the plain role/content dicts an AI endpoint expects.
            Single place where AIMessage objects turn into the plain dicts an endpoint expects.
        """
        return [message.get_message_payload() for message in messages]


    def get_memory_messages_for_ai(self, add_sticky:bool = True, add_reasoning:bool = False, strip_types:"set[str] | None" = None) -> list[dict[str, Any]]:
        """ The conversation MEMORY as the messages array an AI endpoint expects:

                client.chat.completions.create(model=m, messages=conv.get_memory_messages_for_ai())

            Returns only the messages. Generation settings (model, temperature, top_p, max_tokens...)
            are deliberately NOT handled here: this class knows about conversations, not about how a
            given model has to be driven. The calling program assembles the request.

            Args:
                add_sticky: include sticky messages that fall outside the memory window.
                add_reasoning: keep AIC_TYPE_REASONING messages, which are excluded by default.
                strip_types: content types the caller knows the target model cannot accept,
                             e.g. {AIC_TYPE_IMAGE_URL}. See get_memory_messages().

            Internal messages are never included: memory is what the model may see.
        """
        return self.__to_ai_message_array(self.get_memory_messages(add_sticky=add_sticky,
                                                                   add_reasoning=add_reasoning,
                                                                   strip_types=strip_types))


    def get_history_messages_for_ai(self, add_reasoning:bool = False, strip_types:"set[str] | None" = None) -> list[dict[str, Any]]:
        """ The WHOLE conversation as the messages array an AI endpoint expects.

            Same contract as get_memory_messages_for_ai() but ignoring the memory window, so the
            entire conversation is sent. That defeats max_memory and grows without bound: prefer
            get_memory_messages_for_ai() unless you specifically need the full history.

            Internal messages (commands, topic markers) are always excluded: they are a record of
            what the user did, never something to send to a model. Reasoning is excluded too unless
            add_reasoning is True -- it carries ROLE_ASSISTANT, so the role filter alone would let it
            through.
        """
        history_messages:list[AIMessage] = [message for message in self.get_history_messages(ignore_internals=True)
                                            if self.__may_enter_memory(message, add_reasoning)]

        if strip_types:
            history_messages = self.strip_message_types(history_messages, strip_types)

        return self.__to_ai_message_array(history_messages)


    def get_messages(self)-> list[AIMessage]:
        """ The LIVE internal message list - mutating it mutates the conversation, and token counters
            will not follow. Use get_messages_copy(), or get_history_messages(), unless you really
            mean to reach inside.
        """
        return self.__messages

    def get_ith_message(self, i:int) -> AIMessage:
        """ Return the i-th message in the conversation. The system message is at index 0.
            Raise ValueError if the index is out of range.
        """
        length:int = len(self.__messages)
        if i < -length or i >= length:
            raise ValueError("Invalid message index: " + str(i))

        return self.__messages[i]

    def get_ith_message_text(self, n: int = -1) -> str:
        """ Return the n-th message in the chat history.
            If n is -1, return the last message.
            Raise ValueError if the index is out of range.
        """
        return self.get_ith_message(n).get_text()

    def to_string(self, memory_only:bool = False, add_sticky:bool = True, ignore_internals:bool = True) -> str:
        """ ignore_internals is IGNORED when memory_only = True: memory never contains internal messages. """

        result:str = ""
        temp_memory_list:list[AIMessage] = []

        if memory_only:
            temp_memory_list = self.get_memory_messages(add_sticky=add_sticky)
        else:
            temp_memory_list = self.get_history_messages(ignore_internals=ignore_internals)

        for message in temp_memory_list:
            result = result + message.to_string() + "\n"
          
        return result

    def get_messages_copy(self) -> list[AIMessage]:
        """ Returns a copy of the list of messages in the conversation."""
        return self.__messages.copy()

    def count_non_sticky_messages(self) -> int:
        """ Count non sticky messages in the conversation.
        """
        count:int = 0
        for message in self.__messages:
            if not message.is_sticky():
                count += 1
        return count

    def count_sticky_messages(self) -> int:
        """ Count sticky messages in the conversation. 
        """
        count:int = 0
        for message in self.__messages:
            if message.is_sticky():
                count += 1
        return count

    def count_all_messages(self) -> int:
        """ Count all messages in the conversation. 
        """
        return len(self.__messages)

    def get_max_memory_messages(self) -> int:
        return self.__max_memory_messages

    def to_dict(self) -> dict[str, Any]:
        """ Serialise the conversation to plain data: a RECORD of what happened, using the same
            per-message shape that save_conversation() writes. Internal messages are included.

            This is NOT something to send to a model: use get_memory_messages_for_ai() for that.
            (It previously returned an AI request payload, which made it an odd sibling of
            AIMessage.to_dict() and AIMessageContent.to_dict(), both of which serialise state.)

            NOTE: this is a richer view than the on-disk format - save_conversation() writes the
            'messages' entries only, and load_conversation() never reads 'max_memory_messages' back.
            The window is reported here for inspection; it is the caller's to restore.
        """
        return {
            'max_memory_messages': self.__max_memory_messages,
            'messages': [message.to_dict() for message in self.__messages]
        }
    
    def set_max_memory_messages(self, max_memory:int) -> None:
        """ Resize the memory window (message count, excluding the system prompt).
            Nothing is deleted: shrinking simply hides older messages from the model, and growing
            brings them back.

            Raises ValueError if max_memory is not > 0.
        """
        if max_memory <= 0:
            raise ValueError("Max memory must be > 0")
        self.__max_memory_messages = max_memory
        self.recalculate_tokens()
        return

    def remove_nth_message(self, nth:int) -> bool:
        """ Remove the nth message from the conversation.
            Return True if it was removed, False if nth is out of range or is 0:
            SYSTEM_MESSAGE (element [0]) is never removed. Negative indices are not accepted.
        """
        result:bool = nth > 0 and nth < len(self.__messages)
        
        if result:
            self.__messages.pop(nth)
            self.recalculate_tokens()

        return result

    def remove_messages(self, remove_n_messages:int = 1, remove_sticky:bool = True) -> int:
        """ Remove the last n messages from the conversation. 
            If remove_sticky is True, sticky messages will be removed as well.
            raise ValueError if you try to remove <0 messages. 
            SYSTEM_MESSAGE is never removed.
            Return the number of messages actually removed.
        """
        removed_count: int = 0
        if remove_n_messages < 0:
            raise ValueError("You cannot remove a negative number of messages")

        msgs_to_remove: int = min(remove_n_messages, len(self.__messages)-1)
        if msgs_to_remove > 0:

            if remove_sticky:
                self.__messages = self.__messages[:-msgs_to_remove]
                removed_count = msgs_to_remove
            else:
                removed_count = 0
                i = len(self.__messages) - 1
                while i > 0 and removed_count < remove_n_messages:
                    if not self.__messages[i].is_sticky():
                        self.__messages.pop(i)
                        removed_count += 1
                    i -= 1

        self.recalculate_tokens()
        return removed_count

    def remove_non_sticky_messages(self, remove_n_messages:int = 1) -> int:
        """ Remove the last n non-sticky messages from the conversation. 
            SYSTEM_MESSAGE is never removed.        
        """
        removed: int = self.remove_messages(remove_n_messages, False)
        return removed

    def restart(self) -> None:
        """ Remove all messages from the conversation except the system message.
            A conversation always has one, so this never empties the list.
        """
        if self.__messages:
            self.__messages = [self.__messages[0]]
        self.recalculate_tokens()
        return

    def recalculate_tokens(self) -> None:
        """ Recompute every token counter from scratch. Called automatically whenever the message
            list changes, so callers rarely need it - except after editing a message in place via
            AIMessage.set_text(), which this class cannot observe.

            The memory counters are derived from get_memory_messages(add_sticky=True) - the SAME
            selection get_memory_messages_for_ai() sends - so they honour sticky messages outside
            the window, stop at a /NEWTOPIC marker, and exclude internal/reasoning messages exactly
            as the real payload would. A positional "last max_memory entries" cut was tried here
            before and quietly disagreed with what was actually sent whenever sticky messages or a
            /NEWTOPIC marker were in play.
        """
        self.__system_message_tokens = 0    #reset too: a conversation with no system message must
                                            #not keep reporting the previous one's cost
        self.__user_tokens = 0
        self.__assistant_tokens = 0
        self.__total_tokens = 0
        self.__memory_user_tokens = 0
        self.__memory_assistant_tokens = 0
        self.__memory_total_tokens = 0
        self.__biggest_user_msg_tokens = 0
        self.__biggest_assistant_msg_tokens = 0

        #history counters: every message in the conversation.
        for message in self.__messages:
            l_msg_tokens:int = message.get_estimated_tokens()

            if message.get_role() == rc.ROLE_USER:
                self.__user_tokens += l_msg_tokens
                self.__total_tokens += l_msg_tokens
                if l_msg_tokens > self.__biggest_user_msg_tokens:
                    self.__biggest_user_msg_tokens = l_msg_tokens

            elif message.get_role() == rc.ROLE_ASSISTANT:
                self.__assistant_tokens += l_msg_tokens
                self.__total_tokens += l_msg_tokens
                if l_msg_tokens > self.__biggest_assistant_msg_tokens:
                    self.__biggest_assistant_msg_tokens = l_msg_tokens

            elif message.get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                self.__system_message_tokens = l_msg_tokens
                self.__total_tokens += l_msg_tokens

        #memory counters: exactly the messages get_memory_messages() will send to the AI.
        #Counting them here (instead of taking the last __max_memory_messages entries of
        #__messages) keeps the totals in sync with the payload when internal messages,
        #sticky messages outside the memory window or a /NEWTOPIC are in the conversation.
        for message in self.get_memory_messages(add_sticky=True):
            l_msg_tokens = message.get_estimated_tokens()
            self.__memory_total_tokens += l_msg_tokens

            if message.get_role() == rc.ROLE_USER:
                self.__memory_user_tokens += l_msg_tokens
            elif message.get_role() == rc.ROLE_ASSISTANT:
                self.__memory_assistant_tokens += l_msg_tokens
        return

    def get_system_prompt(self) -> str:
        """ The text of message 0, the system prompt. Empty string if the conversation has no
            messages at all (only reachable while loading).
        """
        result:str = ""
        if len(self.__messages) > 0:
            result = self.__messages[0].get_text()
        return result

    def get_messages_count(self, memory_only:bool=False, count_sticky:bool = False, count_internal:bool = True) -> int:
        """
        Return the number of messages in the conversation, including the system message.

        Args:
            memory_only (bool): If True, count only the messages in the memory window; otherwise count
                the whole history.
            count_sticky (bool): If True and memory_only is True, also count sticky messages that fall
                outside the memory window.
            count_internal (bool): If True (the default) internal messages -- recorded commands and
                the /NEWTOPIC marker -- are counted, as they always have been. Only meaningful when
                memory_only is False, since memory never contains them.

                CAREFUL: the history count doubles as the INDEX BOUND for /sticky and /delete, and
                those indices are positions in the full message list. Passing False here gives a
                number smaller than the highest valid index, so do not use it for bounds checking.

        Returns:
            int: The number of messages matching the requested view.
        """
        temp_message_list:list[AIMessage] = []
        result:int           = 0
        if memory_only:
            temp_message_list = self.get_memory_messages(add_sticky=count_sticky)
        else:
            temp_message_list = self.get_history_messages(ignore_internals=not count_internal)

        result = len(temp_message_list)

        return result
    
    def get_memory_messages_no(self) -> int:
        """ Quick upper bound on the memory size: max_memory + 1 for the system prompt, capped by the
            number of messages that exist.

            This is an ARITHMETIC estimate, not a count of the real selection - it ignores sticky
            messages, /NEWTOPIC and internal messages. For the true number use
            len(get_memory_messages()) or get_messages_count(memory_only=True).
        """
        return min(len(self.__messages), self.__max_memory_messages+1)
    
    def get_last_message_text(self) -> str:
        return self.__messages[-1].get_text()
    
    def set_last_message_stickiness(self, sticky:bool) -> None:
        self.__messages[-1].set_sticky(sticky)
        return
    
    def set_nth_message_stickiness(self, nth:int, sticky:bool) -> None:
        """ Set stickiness on message `nth`. Silently does nothing when nth is out of range, or when
            nth is 0: the system prompt is always sticky and is not made otherwise.
        """
        if (nth >= 1) and len(self.__messages) > nth:
            self.__messages[nth].set_sticky(sticky)
        return


    def set_last_two_messages_stickiness(self, sticky:bool) -> None:
        """ Set stickiness on the last two messages - the usual way to pin a question together with
            its answer, so the pair survives the memory window as a unit.
        """
        current_len:int = len(self.__messages)
        if current_len >= 1:
            self.__messages[-1].set_sticky(sticky)
        if current_len >= 2:
            self.__messages[-2].set_sticky(sticky)
        return  
    

    def show_history(self)->int:
        """ Print everything that happened, internal commands included, with full headers. """
        return self.print(memory_only=False, add_sticky=True, add_internal=True, text_only=False)

    def show_memory(self)->int:
        """ Print only what a model would currently be shown, with full headers. """
        return self.print(memory_only=True, add_sticky=True, add_internal=False, text_only=False)

    def show_history_text(self)->int:
        """ show_history() without the type/sticky/model header lines. """
        return self.print(memory_only=False, add_sticky=True, add_internal=True, text_only=True)

    def show_memory_text(self)->int:
        """ show_memory() without the type/sticky/model header lines. """
        return self.print(memory_only=True, add_sticky=True, add_internal=False, text_only=True)

    def print(self, memory_only:bool = False, add_sticky:bool = True, add_internal:bool = False, text_only:bool = False) -> int:
        """
        Print the messages in colour and return how many were actually printed.

        Args:
            memory_only (bool): print the memory window only, instead of the whole history.
            add_sticky (bool): with memory_only, also print sticky messages that fall outside the window.
            add_internal (bool): print internal command messages (/MODEL, /NEWTOPIC...). IGNORED when
                memory_only is True, because memory never contains internal messages.
            text_only (bool): print just role and text, without the type/sticky/model header line.

        Returns:
            int: the number of messages printed.

        Colour follows the role (user, assistant, system, internal). Data URLs are abbreviated to
        their first and last characters so a base64 image does not flood the terminal.
        """

        my_add_internal:bool = add_internal
        if memory_only:
            my_add_internal = False
            temp_memory_list:list[AIMessage] = self.get_memory_messages(add_sticky=add_sticky)
        else:
            temp_memory_list = self.get_history_messages(ignore_internals=not add_internal)

        COLOR_HEADER:str       = "light_cyan"
        COLOR_USER:str         = "light_white"
        COLOR_ASSISTANT:str    = "light_green"
        COLOR_SYSTEM:str       = "light_yellow"
        COLOR_INTERNAL:str     = "gray"

        counter:int = 0
        printed_count:int = 0

        for message in temp_memory_list:

            current_role:str = message.get_role()
            if current_role == rc.ROLE_INTERNAL and not my_add_internal:
                continue

            THE_CONTENT_COLOR:str = COLOR_USER

            if current_role == rc.ROLE_USER:
                THE_CONTENT_COLOR = COLOR_USER
            elif current_role == rc.ROLE_ASSISTANT:
                THE_CONTENT_COLOR = COLOR_ASSISTANT
            elif current_role in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                THE_CONTENT_COLOR = COLOR_SYSTEM
            elif current_role == rc.ROLE_INTERNAL:
                THE_CONTENT_COLOR = COLOR_INTERNAL
            else:
                THE_CONTENT_COLOR = COLOR_USER

            #everything below reads `message`, never temp_memory_list[counter]: counter is a display
            #number that stops advancing on a skip, so using it as an index would print the wrong
            #message for every entry after one
            text_content:str = message.get_text()
            if not text_only:
                model_str:str = f", model={message.get_model()}" if message.get_model() else ""
                printColor(f"{counter}. {message.get_role().upper()} (type={message.get_type()}, sticky={message.is_sticky()}{model_str}):", COLOR_HEADER)
            else:
                printColorSameLine(f"{message.get_role().upper()}:", COLOR_HEADER)

            printColor(f"{text_content}\n", THE_CONTENT_COLOR)

            if message.get_type() == rc.AIC_TYPE_IMAGE_URL:
                image_url: str = message.get_url() or ""
                if image_url.startswith("data:image"):
                    printColor(f"{image_url[:30]}...{image_url[-30:]}", THE_CONTENT_COLOR)
                else:
                    printColor(f"{image_url}", THE_CONTENT_COLOR)
            elif message.get_type() == rc.AIC_TYPE_FILE:
                file_url: str = message.get_file_url() or ""
                if file_url.startswith("data:"):
                    printColor(f"{file_url[:30]}...{file_url[-30:]}", THE_CONTENT_COLOR)
                else:
                    printColor(f"{file_url}", THE_CONTENT_COLOR)
            printed_count += 1
            counter += 1

        return printed_count


    def print_message_No(self, index: int, text_only: bool = False) -> bool:
        """
        Print a specific message by index. Return False if the index-th message does not exist

        Args:
            index (int): The index of the message to print. Negative indices count from the end,
                as in get_ith_message(): -1 is the last message.
            text_only (bool): If True, only print the text content of the message without additional metadata.
        """
        result = False
        if -len(self.__messages) <= index < len(self.__messages):
            message = self.__messages[index]
            content_color = "light_white"  # Default to white
            if message.get_role() == rc.ROLE_USER:
                content_color = "light_white"
            elif message.get_role() == rc.ROLE_ASSISTANT:
                content_color = "light_green"
            elif message.get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                content_color = "light_yellow"

            if not text_only:
                printColor(f"{index} ------------------------------------------ {message.get_role().upper()} (type={message.get_type()}, sticky={message.is_sticky()}):", "light_cyan")
            printColor(f"{message.get_text()}\n", content_color)

            if message.get_type() == rc.AIC_TYPE_IMAGE_URL:
                image_url: str = message.get_url() or ""
                if image_url.startswith("data:image"):
                    printColor(f"{image_url[:30]}...{image_url[-30:]}", content_color)
                else:
                    printColor(image_url, content_color)
            elif message.get_type() == rc.AIC_TYPE_FILE:
                file_url: str = message.get_file_url() or ""
                if file_url.startswith("data:"):
                    printColor(f"{file_url[:30]}...{file_url[-30:]}", content_color)
                else:
                    printColor(file_url, content_color)
            result = True
        else:
            print(f"Invalid message index: {index}. Must be between {-len(self.__messages)} and {len(self.__messages) - 1}.")
        return result

###########################################################
#               MAIN - the tests live in test_AIConversation.py
###########################################################
if __name__ == '__main__':
    #the self test moved to its own file; running this module directly still runs it, so
    #`python AIConversation.py` behaves exactly as it always has
    from test_AIConversation import run_tests
    raise SystemExit(0 if run_tests() else 1)
