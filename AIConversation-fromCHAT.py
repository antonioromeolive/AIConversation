# ------------------------------------------------------------------------------
# File:          AIConversation.py
# Project:       myLocalChat
# Author:        Antonio Romeo
# Version:       1.0.0
# Last modified: 2026-10-10
# Copyright (c) 2024-2026 Antonio Romeo
# License:       MIT - see LICENSE.TXT in the project root
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------

# Description: This module defines the AIConversation class, which represents a conversation between a user and an AI assistant.
# It uses the Azure OpenAI model to extract text from images and writes the extracted text to an output file.

import base64
import json
import mimetypes
import os
import time
from typing import Any
import tiktoken
from AIConversationDB import AIConversationDB
from AIPromptDB import AIPromptDB
import rom_const as rc
from rom_printing_utils import printColor
#import rom_util as ru


class AIMessageContent:
    """ This class represents the content of a message. It can be a text message or an image message.
    """
    def __init__(self, msg_type: str, msg_image_url: "str | None" = "", msg_text: str = "", msg_file_url: "str | None" = "", msg_file_name: "str | None" = None) -> None:
        self.__content:  list[dict[str, Any]] = []
        self.__type: str = msg_type
        self.__image_url: "str | None" = msg_image_url
        self.__file_url: "str | None" = msg_file_url
        self.__file_name: "str | None" = msg_file_name
        self.__text: str = msg_text
        self.set_content(msg_type, msg_image_url, msg_text, msg_file_url, msg_file_name)


    def to_dict(self)-> dict[str, Any]:
        my_dict_version: dict[str, Any] = {
            'type': self.__type,
            'image_url': self.__image_url,
            'file_url': self.__file_url,
            'file_name': self.__file_name,
            'text': self.__text
        }
        return my_dict_version

    @classmethod
    def from_dict(cls, d):
        return cls(d['type'], d.get('image_url', ''), d.get('text', ''), d.get('file_url', ''), d.get('file_name', None))
    
    def get_type(self) -> str:
        return self.__type
    
    def get_image_url(self) -> "str | None":
        return self.__image_url

    def get_file_url(self) -> "str | None":
        return self.__file_url

    def get_file_name(self) -> "str | None":
        return self.__file_name

    def get_text(self) -> str:
        return self.__text
    
    def get_content(self) -> list[dict[str, Any]]:
        return self.__content
    
    def set_content(self, msg_type: str, msg_image_url: "str | None" = "", msg_text: str = "", msg_file_url: "str | None" = "", msg_file_name: "str | None" = None) -> None:
        self.__type: str = msg_type
        self.__image_url: "str | None" = msg_image_url
        self.__file_url: "str | None" = msg_file_url
        self.__file_name: "str | None" = msg_file_name
        self.__text: str = msg_text

        if msg_type == rc.AIC_TYPE_IMAGE_URL:
            msg_image_url = msg_image_url or ""
            if msg_image_url.startswith("http"):
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
            elif msg_image_url.startswith("data:"):
                #ANY data: url is already encoded content, not only data:image. A local file whose type
                #cannot be guessed as an image is stored as data:<other mime>, and set_text() / a reload
                #must not mistake it for a file path to re-read
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
                    #the path is lost once encoded: this is the only place the original name is known
                    self.__file_name = msg_file_name or os.path.basename(msg_file_url)
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
            #the key is added only when a name is known, so a nameless block is exactly what it was before
            if self.__file_name:
                self.__content[0]["file_url"]["filename"] = self.__file_name
        elif msg_type == rc.AIC_TYPE_REASONING:
            #The SEMANTIC type stays "reasoning" - that is what memory filters on and what is saved.
            #The CONTENT BLOCK must be "text": reasoning is the one non-conversational type that can
            #legitimately reach a request (/sendreasoning on), and no endpoint accepts a block of
            #type "reasoning" - the chat API rejects it and messages_to_responses_input() drops it.
            self.__content = [
                {
                    "type": rc.AIC_TYPE_TEXT,
                    "text": msg_text
                }
            ]
        elif msg_type in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC, rc.AIC_TYPE_AGENT]: #store a command, internal message, agent note... will be not sent to AI
            self.__content = [
                {
                    "type": msg_type,
                    "text": msg_text
                }
            ]
        else:
            raise ValueError("AIMessageContent.set_content - Invalid message type: " + msg_type)
        
        
    def to_string(self) -> str:
        result:str= ""
        if self.__type == rc.AIC_TYPE_IMAGE_URL:
            result = f"Image URL: {self.__image_url}\nText: {self.__text}"
        elif self.__type == rc.AIC_TYPE_FILE:
            result = f"File  URL: {self.__file_url}\nText: {self.__text}"
        elif self.__type == rc.AIC_TYPE_TEXT:
            result = f"Text: {self.__text}"
        elif self.__type in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC, rc.AIC_TYPE_AGENT]:
            result = f"Text: {self.__text}"
        return result


class AIMessage:
    def __init__(self, role: str, content_text:str, msg_type:str, content_image_url:"str | None" = None, sticky:bool = False, content_file_url:"str | None" = None, model:str = "", content_file_name:"str | None" = None) -> None:
        """ This a message to or from AI assistant. It can be a text message or an image message.
        Args:   role: str - the role of the message. It can be "User" or "System" or "Assistant" - MUST NOT be None
                msg_type - the type of the message. It can be "text" or "image_url" - MUST NOT be None
                content_text: str - the text of the message - MUST NOT be None
                content_image_url: str - the url of the image to be sent. if not None the message contains a link to an image
                sticky: bool - if the message is sticky or not
                model: str - the name of the AI model this message was sent to / produced by (optional, defaults to "")
                content_file_name: str - original name of the attached file (optional, derived from a local path when not given)
                NOTE: once built you can change the content_text, content_image_url and sticky flag using the appropriate methods below
                but it requires you to pass an AIMessageContent object to the set_content method.
                the get_message_payload() method can be used to build a conversation string to be passed to (Azure) OpenAI
        """
        self.__role: str = role
        self.__content: AIMessageContent
        self.__sticky: bool = sticky
        self.__model: str = model
        self.__estimated_tokens: int = 0
        self.__image_tokens: int = 0    #token cost of the image payload only, fixed at construction. See __refresh_estimated_tokens()
        self.__is_internal: bool = False

        #The (role, type) contract lives in rc.AIC_ALLOWED_TYPES_BY_ROLE and is checked here and nowhere
        #else. Three separate failures, each with its own message:
        #  - unknown role: no row in the table (f-string, not concatenation, so a corrupt saved message
        #    carrying role=None is REPORTED rather than raising TypeError inside the error path);
        #  - unknown type: not a known content type at all;
        #  - illegal pair: a known role carrying a type its row forbids, e.g. an internal message with
        #    an image. Enforced on construction AND on load (from_dict comes through here), so a saved
        #    file holding an illegal pair is refused rather than trusted.
        #a None text becomes "": an image may legitimately have no caption, and the types that do require
        #text still refuse an empty one below
        if content_text is None:
            content_text = ""

        allowed_types = rc.AIC_ALLOWED_TYPES_BY_ROLE.get(role)
        if allowed_types is None:
            raise ValueError(f"AIMessage.__init__ - Invalid message role: {role!r}")

        if (msg_type not in [rc.AIC_TYPE_TEXT, rc.AIC_TYPE_IMAGE_URL, rc.AIC_TYPE_FILE, rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_REASONING, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC, rc.AIC_TYPE_AGENT]):
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
        elif (msg_type in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC]):
            if content_text is None or content_text == "":
                raise ValueError("Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(msg_type, None, content_text)
            self.__is_internal = True
        elif (msg_type == rc.AIC_TYPE_REASONING):
            if content_text is None or content_text == "":
                raise ValueError("Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_REASONING, None, content_text)
            self.__is_internal = True
        elif (msg_type == rc.AIC_TYPE_AGENT):
            if content_text is None or content_text == "":
                raise ValueError("Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_AGENT, None, content_text)
            self.__is_internal = True
        elif (msg_type == rc.AIC_TYPE_TEXT):
            if content_text is None or content_text == "":
                raise ValueError("AIMessage.__init__ - Missing TEXT for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_TEXT, None, content_text)
        elif (msg_type == rc.AIC_TYPE_FILE):
            if content_file_url is None or content_file_url == "":
                raise ValueError("Missing File URL for " + msg_type + " message")
            self.__content = AIMessageContent(rc.AIC_TYPE_FILE, msg_file_url=content_file_url, msg_text=content_text, msg_file_name=content_file_name)
        else:
            raise ValueError("AIMessage.__init__ - Invalid message type: " + msg_type)

        self.__refresh_estimated_tokens()

    def __refresh_estimated_tokens(self) -> None:
        """ Recompute the token estimate from the CURRENT content. Used at construction and by
            set_text(), so an edited message no longer keeps the token count of its old text.
            App generated messages (INTERNAL, WARNING, NEWTOPIC) are never sent to AI so they cost 0.
            The image cost is NOT recomputed here: it is derived once in __init__ from the url as it
            was originally passed in, because AIMessageContent normalises a local path into a much
            longer base64 data url and re-measuring it would change the estimate on a text edit.
        """
        if self.__content.get_type() in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC]:
            self.__estimated_tokens = 0 #not sent to AI so no token no cry
        else:
            self.__estimated_tokens = self.__image_tokens + self.num_tokens_from_string(self.__content.get_text())
        return

    def num_tokens_from_picture(self, encoded_image:str) -> int:
        """ I need to understand better the way to calculate this... 
            in the meanwhile this is the expected tokens of a 1024x1024 image 
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
        """return num of tokens from a string... model name sample: gpt-4 """
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
        my_dict_version: dict[str, Any] = {
            'role': self.__role,
            'msg_type': self.__content.get_type(),
            'content_text': self.__content.get_text(),
            'content_image_url': self.__content.get_image_url(),
            'content_file_url': self.__content.get_file_url(),
            'content_file_name': self.__content.get_file_name(),
            'sticky': self.__sticky,
            'model': self.__model
        }
        return my_dict_version
    
    def to_string(self) -> str:
        result:str= f"Role: {self.__role}\nContent: {self.__content.to_string()}\nSticky: {self.__sticky}"
        return result


    @classmethod
    def from_dict(cls, d):
        return cls(
            role = d.get('role'),
            content_text = d.get('content_text', ""),
            msg_type = d.get('msg_type', rc.AIC_TYPE_TEXT),
            content_image_url = d.get('content_image_url', None),
            sticky = bool(d.get('sticky', False)),
            content_file_url = d.get('content_file_url', None),
            model = d.get('model', ''),
            content_file_name = d.get('content_file_name', None)
        )
        
    def get_type(self) -> str:
        return self.__content.get_type()
    
    def get_url(self) -> "str | None":
        return self.__content.get_image_url()

    def get_file_url(self) -> "str | None":
        return self.__content.get_file_url()

    def get_file_name(self) -> "str | None":
        return self.__content.get_file_name()
    
    def get_text(self) -> str:
        return self.__content.get_text()
    
    def set_text(self, text: str) -> None:
        """ Replace the text of the message and refresh its token estimate.
            NOTE: callers holding this message inside an AIConversation should call
            AIConversation.recalculate_tokens() afterwards so the conversation totals follow.
        """
        self.__content.set_content(self.__content.get_type(), self.__content.get_image_url(), text, self.__content.get_file_url(), self.__content.get_file_name())
        self.__refresh_estimated_tokens()

    def get_role(self) -> str:
        return self.__role

    def set_role(self, role: str) -> None:
        """ Change the role of the message, subject to the same (role, type) contract the constructor
            enforces: the new role must exist and must be allowed to carry this message's content type.
            Raises ValueError, leaving the message untouched, when the pair is not permitted.

            NOTE: callers holding this message inside an AIConversation should call
            AIConversation.recalculate_tokens() afterwards, as the user/assistant token split follows the role.
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

    def set_sticky(self, sticky: bool) -> None:
        """ Sticky messages are not removed from the conversation memory when a new message is added. 
            They are kept in the conversation until the user decides to remove them. 
            However there is a flag in the removemessages method to remove sticky messages as well.
        """
        self.__sticky = sticky

    def is_sticky(self) -> bool:
        return self.__sticky

    def is_internal(self) -> bool:
        return self.__is_internal

    def set_content(self, content: AIMessageContent) -> None:
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
        payload = {
            "role": self.get_role(),
            #"sticky": self.is_sticky(),
            "content": self.get_content().get_content()
        }
        return payload


class AIConversation:
    def __init__(self, system_message:"str | None" = None,  max_memory:int = rc.AIC_DEFAULT_MEMORY_SIZE) -> None:
        """ This class represents a conversation between a user and an AI assistant.
            Careless of the length of the conversation only "max_memory" messages will be used from the get_memory_messages_for_ai() method.
            Extra messages will be ignored...
        """
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
            supplies the model and the temperature. Pass the same max_memory you would pass to the
            constructor; the default matches the constructor's.
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
        return self.__memory_total_tokens

    def get_total_tokens(self) -> int:
        return self.__total_tokens

    def get_memory_assistant_tokens(self) -> int:
        return self.__memory_assistant_tokens
    
    def get_user_tokens(self) -> int:
        return self.__user_tokens
    
    @staticmethod
    def _migrate_legacy_message(role:str, msg_type:str, text:str) -> "tuple[str, str, bool]":
        """ Bring a message saved by an older version up to the current (role, type) contract.

            Returns (role, msg_type, was_migrated). Applied by BOTH loaders (load_conversation and
            load_from_db) so a conversation behaves the same whichever store it came from, and applied
            BEFORE the AIMessage is constructed - an illegal pair would otherwise be refused, caught,
            and the message dropped.

            This matters more than it looks. Older builds validated the role and the type SEPARATELY,
            so every combination was accepted and could reach a saved file. Anything the current
            contract refuses has to be repaired here or it is silently lost on load.

            NAMED REWRITES, for the two pairings this application really produced:

              internal + reasoning              -> assistant + reasoning
                  Reasoning was filed under ROLE_INTERNAL. It is assistant output, and memory now
                  excludes it by TYPE rather than by role.

              internal + internal, text "/NEWTOPIC" -> internal + newtopic
                  The topic marker was an ordinary internal message recognised by its text. Without
                  this rewrite an old conversation loses every topic boundary and the model is fed
                  everything back to the first message.

            GENERAL REPAIR, for any other pair the contract refuses:

              a bookkeeping type on a non-internal role  -> move it to ROLE_INTERNAL
              reasoning on any role but assistant        -> move it to ROLE_ASSISTANT
              any other type on ROLE_INTERNAL            -> it is a recorded command: type INTERNAL
              anything else, where the role accepts text -> degrade the type to TEXT, keeping the
                  message text. An attachment the role may not carry is dropped, but the words
                  survive; losing a whole message (a system prompt, say) would be far worse.
        """
        #named rewrites first: they are more specific than the general repair below
        if role == rc.ROLE_INTERNAL and msg_type == rc.AIC_TYPE_REASONING:
            return rc.ROLE_ASSISTANT, msg_type, True

        #an agent note on a role that may not carry it (system, developer, internal) is kept as an internal
        #note: the general repair below could otherwise degrade it to TEXT and turn a record into dialogue
        if msg_type == rc.AIC_TYPE_AGENT and msg_type not in rc.AIC_ALLOWED_TYPES_BY_ROLE.get(role, set()):
            return rc.ROLE_INTERNAL, rc.AIC_TYPE_INTERNAL, True

        if role == rc.ROLE_INTERNAL and msg_type == rc.AIC_TYPE_INTERNAL and (text or "").strip().upper() == rc.AIC_COMMAND_NEWTOPIC:
            return role, rc.AIC_TYPE_NEWTOPIC, True

        allowed_types = rc.AIC_ALLOWED_TYPES_BY_ROLE.get(role)
        if allowed_types is None or msg_type in allowed_types:
            return role, msg_type, False        #unknown role, or already legal: nothing to repair

        if msg_type in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC]:
            return rc.ROLE_INTERNAL, msg_type, True

        if msg_type == rc.AIC_TYPE_REASONING:
            return rc.ROLE_ASSISTANT, msg_type, True

        if role == rc.ROLE_INTERNAL:
            #an internal message carrying anything else (older builds allowed plain text here) is a
            #recorded command: keep the text, give it the type the role does accept
            return role, rc.AIC_TYPE_INTERNAL, True

        if rc.AIC_TYPE_TEXT in allowed_types:
            return role, rc.AIC_TYPE_TEXT, True

        return role, msg_type, False            #nothing sensible to do: let the loader report it


    def load_conversation(self, full_path_name:str) -> None:
        """ Replace this conversation with the one stored in full_path_name (JSON Lines).

            Nothing is destroyed until the file has been read: the messages are built into a separate
            list and only swapped in at the end, so a file that cannot be opened or read leaves this
            conversation exactly as it was.

            A line that fails is REPORTED AND SKIPPED rather than aborting the load: unparseable JSON,
            a missing required key or content a message type rejects. Blank lines are ignored.
        """
        loaded: list[AIMessage] = []
        line_number:int = 0
        migrated_count:int = 0

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

                    #migrate BEFORE constructing: an old (role, type) pair would be refused by the
                    #contract, and the message would be skipped instead of upgraded
                    new_role, new_type, was_migrated = self._migrate_legacy_message(
                        message_dict.get('role'), message_dict.get('msg_type', rc.AIC_TYPE_TEXT), message_dict.get('content_text', ""))
                    if was_migrated:
                        message_dict['role']     = new_role
                        message_dict['msg_type'] = new_type
                        migrated_count += 1

                    new_message = AIMessage.from_dict(message_dict)
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

        if migrated_count:
            printColor(f"AIConversation.load_conversation() - {migrated_count} message(s) upgraded from an older format "
                       f"(reasoning moved to the assistant role, /NEWTOPIC given its own type). "
                       f"Save the conversation to keep the new format.", "yellow")

        #message 0 is always the system prompt: restart() and get_system_prompt() rely on it, so a
        #file that is empty, truncated or simply does not start with one gets the default
        if not loaded or loaded[0].get_role() not in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
            printColor(f"AIConversation.load_conversation() - no system message found in {full_path_name}, using the default prompt.", "yellow")
            loaded.insert(0, AIMessage(rc.ROLE_SYSTEM, rc.AIC_DEFAULT_SYSTEM_PROMPT, rc.AIC_TYPE_TEXT, None, True))

        self.__restart_internal()
        self.__messages = loaded
        self.recalculate_tokens()
        return

    def save_conversation(self, full_path_name:str, max_retries:int = 3) -> None:
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
        """ Draw a line under the conversation so far: append the AIC_TYPE_NEWTOPIC marker.

            From here on memory stops at this marker, so the model sees the system prompt plus only
            what follows it. Nothing is deleted: earlier messages stay in history, are still saved and
            are still shown by print().

            The marker has its OWN TYPE rather than being an AIC_TYPE_INTERNAL message recognised by
            its text. Text matching was fragile: any recorded note beginning with "/NEWTOPIC" looked
            like a marker without being one, and case differences could hide a real one. Now the only
            way to create a marker is to call this method.
        """
        self.add_message(rc.ROLE_INTERNAL, rc.AIC_COMMAND_NEWTOPIC, rc.AIC_TYPE_NEWTOPIC)
        return
    
    def add_comment(self, text:str) -> None:
        """ Record a note in the conversation: something that HAPPENED rather than something that was
            said. Stored as ROLE_INTERNAL + AIC_TYPE_INTERNAL, so it appears in the history and in the
            saved file, is never sent to a model, and costs no tokens.

            The application decides what is worth recording - a parameter change ("/MODEL:gpt-4o"), or
            any free text. The text is stored VERBATIM: this class does not parse it, whitelist it or
            change its case.

            (This used to be add_command(), which only stored strings beginning with one of six known
            prefixes and silently discarded everything else - so forgetting to register a new command
            in a list on the other side of the codebase made it vanish without a word. The class no
            longer needs that check: the one internal message it actually interprets is the topic
            marker, and new_topic() writes it with its own AIC_TYPE_NEWTOPIC.)
        """
        self.add_message(rc.ROLE_INTERNAL, text, rc.AIC_TYPE_INTERNAL)
        return


    def add_message(self, msg_role:str, msg_text:str, msg_type:str=rc.AIC_TYPE_TEXT, image_url:"str | None" = None, is_sticky = False, file_url:"str | None" = None, model:str = "", file_name:"str | None" = None) -> None:
        """ You cannot add a system message after the first message has been added to the conversation.
            You can add as many user and assistant messages as you want.
            A system message will replace the one at the beginning of the conversation.
            Raise an exception for unsupported message types

            The (role, type) contract itself is NOT re-checked here: AIMessage enforces
            rc.AIC_ALLOWED_TYPES_BY_ROLE in its constructor, and duplicating the list of valid types in
            this method is exactly how it fell out of step with rom_const before.
        """
        #the prompt is always sticky, like the one installed by the constructor and change_system_message()
        is_still_sticky:bool = is_sticky or msg_role in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]

        new_msg = AIMessage(msg_role, msg_text, msg_type, image_url, is_still_sticky, file_url, model, file_name)

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

            Internal messages never can: commands, warnings and the topic marker are things the
            application did, not conversation. Both role and type are checked - the constructor now
            refuses to pair an internal type with any other role, but a conversation saved before that
            rule existed can still hold one.

            Reasoning only when explicitly asked for. It is assistant output, but it records HOW an
            answer was reached rather than what was said, so it belongs to the record, not the dialogue.
        """
        result = True
        if message.get_role() == rc.ROLE_INTERNAL or message.get_type() in [rc.AIC_TYPE_INTERNAL, rc.AIC_TYPE_WARNING, rc.AIC_TYPE_NEWTOPIC]:
            result = False

        if message.get_type() == rc.AIC_TYPE_REASONING and not add_reasoning:
            result = False

        #agent notes are a record of what a pre-check / post-check agent replaced: never sent, no flag to change it
        if message.get_type() == rc.AIC_TYPE_AGENT:
            result = False

        return result


    def get_memory_messages(self, add_sticky:bool = True, add_reasoning:bool = False, strip_types:"set[str] | None" = None) -> list[AIMessage]:
        """
        Retrieve the list of messages that are part of the conversation memory.

        This method returns a subset of the conversation messages that fit within the memory limit (`__max_memory_messages`).
        It includes sticky messages and handles special cases, such as detecting a new topic.

        The function works as follows:
        - Iterates through the messages in reverse order (starting from the most recent).
        - Adds messages to the memory list until the memory limit is reached.
        - Includes sticky messages even if they exceed the memory limit.
        - Stops adding messages if an AIC_TYPE_NEWTOPIC marker is encountered.
        - Ensures the system message (first message in the conversation) is included if a new topic is found.

        Internal messages (commands, warnings, the topic marker) are ALWAYS excluded: they are things the
        application did, never something to send. Reasoning messages are excluded too unless add_reasoning
        is True -- reasoning is assistant output, but it records HOW an answer was reached rather than what
        was said. Neither kind consumes a slot in the window when excluded, so hiding them makes room for
        more real conversation instead of wasting the budget.

        Args:
            add_sticky (bool): include sticky messages that fall outside the memory window. This EXPANDS
                the selection: it reaches outside the window and pulls messages back in.
            add_reasoning (bool): keep AIC_TYPE_REASONING messages. Unlike add_sticky this never expands
                the selection: it only stops excluding reasoning that is already inside the window, so a
                sticky reasoning message outside the window is still left out.
            strip_types (set[str] | None): the rc.AIC_TYPE_* values the CALLER knows the target model
                cannot accept. Only AIC_TYPE_IMAGE_URL and AIC_TYPE_FILE are meaningful -- they are the
                only types carrying an attachment. Matching messages are replaced by a text-only
                stand-in carrying their original text plus the placeholder from AIC_PLACEHOLDER_BY_TYPE;
                None or an empty set strips nothing, and any other value is never matched.
                Deciding what a model can accept is the caller's job: this class holds no model
                knowledge and never guesses. See AIModel.get_strip_types(), which builds the set from
                the model's own capability flags.

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

            if message.get_type() == rc.AIC_TYPE_NEWTOPIC:
                new_topic_found = True
                break
            elif message.get_role() in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                system_message_found = True
                new_topic_found      = True
                temp_memory_list.append(message)
                break

            if len(temp_memory_list) < self.__max_memory_messages:
                #inside the window. Messages that may not enter memory are skipped WITHOUT using up a
                #window slot, so hiding them buys room for real conversation instead of wasting it
                if self.__may_enter_memory(message, add_reasoning):
                    temp_memory_list.append(message)
                continue
            else:
                #outside the window only STICKY messages are pulled back in, and never reasoning:
                #add_reasoning keeps reasoning already inside the window, it does not widen it
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

            Lets a caller report "2 images omitted" BEFORE sending, instead of stripping silently.
            Counts what memory would actually contain, so internal messages never appear and
            reasoning appears only when asked for.
        """
        counts: dict[str, int] = {}

        for message in self.get_memory_messages(add_sticky=add_sticky, add_reasoning=add_reasoning):
            counts[message.get_type()] = counts.get(message.get_type(), 0) + 1

        return counts


    def get_history_messages(self, ignore_internals:bool = False) -> list[AIMessage]:
        """
        Retrieve the FULL conversation history (no memory limit, no sticky/new-topic logic).

        This is the human facing counterpart of get_memory_messages(): every message ever added to
        the conversation, in insertion order, starting with the system message at index 0.

        Args:
            ignore_internals (bool): If True, ROLE_INTERNAL messages (commands such as /MODEL, warnings
                and the /NEWTOPIC marker) are filtered out. Default is False, i.e. the history shows
                everything -- internal messages are part of the record of what happened.

        Reasoning messages are ALWAYS kept here whatever ignore_internals says: they carry ROLE_ASSISTANT
        and are a real part of the record. It is get_history_messages_for_ai() that drops them, because
        that view is a request payload rather than a record.

        Returns:
            list[AIMessage]: When ignore_internals is False the internal list is returned as-is
            (do not mutate it); when True a new filtered list is returned.
        """
        if not ignore_internals:
            return self.__messages

        return [message for message in self.__messages if message.get_role() != rc.ROLE_INTERNAL]


    @staticmethod
    def __to_ai_message_array(messages:list[AIMessage]) -> list[dict[str, Any]]:
        """ Single place where AIMessage objects turn into the plain role/content dicts an endpoint expects. """
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
                strip_types: the rc.AIC_TYPE_* values the caller knows the target model cannot
                             accept. Only two are meaningful, because only these carry an
                             attachment that a text-only model chokes on:

                                 rc.AIC_TYPE_IMAGE_URL   an image
                                 rc.AIC_TYPE_FILE        a file attachment

                             Each matching message is replaced by a text-only stand-in holding its
                             original text plus the placeholder from rc.AIC_PLACEHOLDER_BY_TYPE.
                             None or an empty set strips nothing.

                             Any other value is simply never matched, so passing it is harmless but
                             pointless: AIC_TYPE_TEXT is what stripping produces, and the internal
                             types never reach memory in the first place.

                             The caller decides what to pass -- this class holds no model knowledge.
                             AIModel.get_strip_types() builds the set from a model's own capability
                             flags, and AIModel.get_messages_for_ai() passes it in for you.

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

            Internal messages (commands, warnings, topic markers) are always excluded: they are a
            record of what the application did, never something to send to a model. Reasoning is
            excluded too unless add_reasoning is True -- it carries ROLE_ASSISTANT, so the role filter
            alone would let it through.
        """
        history_messages:list[AIMessage] = [message for message in self.get_history_messages(ignore_internals=True)
                                            if self.__may_enter_memory(message, add_reasoning)]

        if strip_types:
            history_messages = self.strip_message_types(history_messages, strip_types)

        return self.__to_ai_message_array(history_messages)

    def get_messages(self)-> list[AIMessage]:
        return self.__messages

    def get_ith_message(self, i:int) -> AIMessage:
        """ Return the i-th message in the conversation. The system message is at index 0.
            Raise ValueError if the index is out of range.
        """
        length:int = len(self.__messages)
        if i < -length or i >= length:
            raise ValueError("Invalid message index: " + str(i))

        return self.__messages[i]

    def to_string(self, memory_only:bool = False, add_sticky:bool = True, ignore_internals:bool = False) -> str:
        """ The conversation as one human readable string.

            Args:
                memory_only: render the memory window instead of the whole history.
                add_sticky: with memory_only, include sticky messages from outside the window.
                ignore_internals: drop internal messages (commands, warnings, the topic marker).
                    Default False, i.e. the string shows everything, as it always has.
                    IGNORED when memory_only is True: memory never contains internal messages.
        """
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
            per-message shape save_conversation() writes. Internal messages are included.

            This is NOT something to send to a model - use get_memory_messages_for_ai() for that.
            (It used to return an AI request payload with hardcoded temperature/top_p/max_tokens,
            which made it an odd sibling of AIMessage.to_dict() and AIMessageContent.to_dict(), both
            of which serialise state, and left generation settings owned by a class that owns none.)

            NOTE: richer than the on-disk format - save_conversation() writes the 'messages' entries
            only, and load_conversation() never reads 'max_memory_messages' back. The window is
            reported here for inspection; restoring it is the caller's job, see from_file().
        """
        return {
            'max_memory_messages': self.__max_memory_messages,
            'messages': [message.to_dict() for message in self.__messages]
        }
    
    def set_max_memory_messages(self, max_memory:int) -> None:
        if max_memory <= 0:
            raise ValueError("Max memory must be > 0")
        self.__max_memory_messages = max_memory
        self.recalculate_tokens()
        return

    def remove_nth_message(self, nth:int) -> bool:
        """ Remove the nth message from the conversation. 
            Return False if nth is out of range or zero.
            SYSTEM_MESSAGE (history message 0) can NEVER be deleted.
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
            SYSTEM_MESSAGE (history message 0) can NEVER be deleted.
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
        #Reasoning is skipped explicitly: it carries ROLE_ASSISTANT, so without this it would be added
        #to the assistant and total counters. It is never sent as part of memory, so counting it would
        #inflate every displayed total with text no model ever receives.
        #Agent notes are skipped for the same reason: they carry ROLE_USER / ROLE_ASSISTANT but are never sent.
        for message in self.__messages:
            if message.get_type() in [rc.AIC_TYPE_REASONING, rc.AIC_TYPE_AGENT]:
                continue

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
            count_internal (bool): If True (the default) internal messages -- recorded commands,
                warnings and the /NEWTOPIC marker -- are counted, as they always have been. Only
                meaningful when memory_only is False, since memory never contains them.

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

    def get_last_message_text(self) -> str:
        return self.__messages[-1].get_text()
    
    def set_last_message_stickiness(self, sticky:bool) -> None:
        self.__messages[-1].set_sticky(sticky)
        return
    
    def set_nth_message_stickiness(self, nth:int, sticky:bool) -> None:

        if (nth >= 1) and len(self.__messages) > nth:
            self.__messages[nth].set_sticky(sticky)   
        return


    def set_last_two_messages_stickiness(self, sticky:bool) -> None:
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
        Print the whole list of messages and return the number of elements actually printed.

        Args:
            memory_only (bool): If True, only print messages that are within the memory limit.
            add_sticky (bool): If True and memory_only is True, include sticky messages in the print (even if not in the memory limit).
            add_internal (bool): If True, print internal messages (recorded commands, warnings, the
                /NEWTOPIC marker). IGNORED when memory_only is True: memory never contains them.
            text_only (bool): If True, only print the text content of the messages without additional metadata.

        Returns:
            int: The number of messages actually printed.

        The method iterates through the list of messages and prints them based on the specified conditions.
        It uses ANSI escape codes to color the output based on the role of the message (user, assistant, system).
        If the message type is an image URL, it prints the URL differently based on its format.

        NOTE: this replaced the old print2(). The name was previously occupied by a do-nothing
        deprecation stub, which is why the real method had to be called print2.
        """
    
        """
        RU_COLOR_RED:str    = "\033[91m {}\033[00m"
        RU_COLOR_DARK_GREEN:str   = "\033[32m {}\033[00m"
        RU_COLOR_BLUE:str   = "\033[94m {}\033[00m"
        RU_COLOR_PINK:str   = "\033[95m {}\033[00m"
        RU_COLOR_BLACK:str  = "\033[98m {}\033[00m"
        RU_COLOR_GRAY:str   = "\033[99m {}\033[00m"
        """ 

        my_add_internal:bool = add_internal
        if memory_only:
            my_add_internal = False
            temp_memory_list:list[AIMessage] = self.get_memory_messages(add_sticky=add_sticky)
        else:
            temp_memory_list = self.__messages

        RU_COLOR_YELLOW:str       = rc.RU_COLOR_YELLOW
        RU_COLOR_CYAN:str         = rc.RU_COLOR_CYAN
        RU_COLOR_WHITE:str        = rc.RU_COLOR_LIGHT_WHITE
        RU_COLOR_LIGHT_GREEN:str  = rc.RU_COLOR_LIGHT_GREEN
        RU_COLOR_GRAY:str         = rc.RU_COLOR_GRAY
        RU_COLOR_ORANGE:str       = rc.RU_COLOR_ORANGE
        

        history_indices = {id(message): index for index, message in enumerate(self.__messages)}
        printed_count:int = 0

        for message in temp_memory_list:
            
            current_role:str = message.get_role()
            if current_role == rc.ROLE_INTERNAL and not my_add_internal:
                continue

            THE_CONTENT_COLOR:str = RU_COLOR_WHITE

            if message.get_type() in [rc.AIC_TYPE_REASONING, rc.AIC_TYPE_AGENT]:
                THE_CONTENT_COLOR = RU_COLOR_GRAY
            elif current_role == rc.ROLE_USER:
                THE_CONTENT_COLOR = RU_COLOR_WHITE
            elif current_role == rc.ROLE_ASSISTANT:
                THE_CONTENT_COLOR = RU_COLOR_LIGHT_GREEN
            elif current_role in [rc.ROLE_SYSTEM, rc.ROLE_DEVELOPER]:
                THE_CONTENT_COLOR = RU_COLOR_YELLOW
            elif current_role == rc.ROLE_INTERNAL:
                THE_CONTENT_COLOR = RU_COLOR_ORANGE if message.get_type() == rc.AIC_TYPE_WARNING else RU_COLOR_GRAY
            else:
                THE_CONTENT_COLOR = RU_COLOR_WHITE

            # Preserve history indices even when memory or internal-message filters leave gaps.
            counter = history_indices[id(message)]
            text_content:str = message.get_text()
            if not text_only:
                model_str:str = f", model={message.get_model()}" if message.get_model() else ""
                print(RU_COLOR_CYAN.format(f"{counter}. {message.get_role().upper()} (type={message.get_type()}, sticky={message.is_sticky()}{model_str}):"))
            else:
                print(RU_COLOR_CYAN.format(f"{message.get_role().upper()}:"), end="")

            print(THE_CONTENT_COLOR.format(f"{text_content}\n"))

            if message.get_type() == rc.AIC_TYPE_IMAGE_URL:
                image_url: str = message.get_url() or ""
                if image_url.startswith("data:image"):
                    print(THE_CONTENT_COLOR .format(f"{image_url[:30]}...{image_url[-30:]}"))
                else:
                    print(THE_CONTENT_COLOR .format(f"{image_url}"))
            elif message.get_type() == rc.AIC_TYPE_FILE:
                file_url: str = message.get_file_url() or ""
                if file_url.startswith("data:"):
                    print(THE_CONTENT_COLOR .format(f"{file_url[:30]}...{file_url[-30:]}"))
                else:
                    print(THE_CONTENT_COLOR .format(f"{file_url}"))
            printed_count += 1
            
        return printed_count


    def print_specific_message(self, index: int, text_only: bool = False) -> bool:
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
            if message.get_type() in [rc.AIC_TYPE_REASONING, rc.AIC_TYPE_AGENT]:
                content_color = "gray"
            elif message.get_role() == rc.ROLE_USER:
                content_color = "light_white"
            elif message.get_role() == rc.ROLE_ASSISTANT:
                content_color = "light_green"
            elif message.get_role() == rc.ROLE_SYSTEM:
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


    def load_from_db(self, db_filename:str, chat_name:str)->None:
        """ Load the conversation from a SQLite database file.
        """
        self.__restart_internal()

        my_db = None
        try:
            my_db = AIConversationDB(db_filename)
            rows = my_db.get_chat(chat_name)
        finally:
            if my_db:
                my_db.close()

        i:int = 0
        migrated_count:int = 0
        for line in rows:
            #            print(f"Line {i}: {str(line)}")
            try:
                # SELECT * column order (from conversations table):
                # line[0] = chat          (e.g. 'z_TEST - Agent Reply with Counter.hp')
                # line[1] = model         (e.g. 'gpt-o')
                # line[2] = msg_number    (e.g. 0)
                # line[3] = role          (e.g. 'system')
                # line[4] = msg_type      (e.g. 'text')
                # line[5] = content_text  (e.g. 'You are a test agent...')
                # line[6] = content_image_url (e.g. None)
                # line[7] = sticky        (e.g. 1)
                # line[8] = created_at    (e.g. '2024-11-27 22:22:35')
                # line[9] = content_file_name (e.g. 'report.pdf' - None for non file rows and older saves)

                #add_message(self, msg_role:str, msg_text:str, msg_type:str=rc.AIC_TYPE_TEXT, image_url:str = None, is_sticky = False, file_url:str = None)
                l_msg_type = line[4]
                l_url      = line[6]
                l_model    = line[1]
                l_file_name = line[9] if len(line) > 9 else None

                #same upgrade the file loader applies, so a conversation behaves identically whichever
                #store it came from. Without it an old internal+reasoning row would be refused by the
                #(role, type) contract, caught below, and silently dropped
                l_role, l_msg_type, was_migrated = self._migrate_legacy_message(line[3], l_msg_type, line[5])
                if was_migrated:
                    migrated_count += 1

                if l_msg_type == rc.AIC_TYPE_FILE:
                    self.add_message(l_role, line[5], l_msg_type, None, bool(line[7]), l_url, l_model, l_file_name)
                else:
                    self.add_message(l_role, line[5], l_msg_type, l_url, bool(line[7]), model=l_model)
                i+=1
            except ValueError as e:
                print(f"AIConversation.load_from_db() - Error loading message {i}: {e}")
                print(f"Wrong Line {i}: {str(line)}")
                continue
            except Exception as e:
                raise e

        if migrated_count:
            printColor(f"AIConversation.load_from_db() - {migrated_count} message(s) upgraded from an older format "
                       f"(reasoning moved to the assistant role, /NEWTOPIC given its own type). "
                       f"Save the conversation to keep the new format.", "yellow")
        return

    def save_prompt_to_db(self, db_filename:str, chat_name:str, chat_model:str="", db:"AIPromptDB|None"=None)->None:
        """ Save only the prompt of a conversation to a SQLite database file.
            If an open `db` connection is supplied it is reused (and left open) so a caller
            can save many prompts over a single connection; otherwise one is opened and closed here.
        """
        my_db = db
        owns_db:bool = db is None
        try:
            if my_db is None:
                my_db = AIPromptDB(db_filename)
            # Insert a row of data

            the_role:str = self.__messages[0].get_role()

            if the_role == rc.ROLE_SYSTEM or the_role == rc.ROLE_DEVELOPER or the_role == rc.ROLE_ASSISTANT:
                my_db.insert_prompt(self.__messages[0].get_text(), chat_name, chat_model)
            else:
                print(f"AIConversation.save_prompt_to_db() - Error saving prompt: The first message is not a system message. Role is: {the_role}")

        except FileNotFoundError:
            # Handle the error, e.g., create a new database or exit
            print(f"Data NOT Saved to DB. Database file not found: {db_filename}")
            print("Check the property file to change current db_filename settings. Note the WHOLE path need to exists (folders will not be created)!")
            raise

        except Exception as e:
            print(f"Prompt Data NOT Saved to DB. An error occurred: {e}")
            raise

        finally:
            if owns_db and my_db:
                my_db.close()  # Ensure connection is closed
        return

    def save_to_db(self, db_filename:str, chat_name:str, chat_model:str="", db:"AIConversationDB|None"=None)->None:
        """ Save the conversation to a SQLite database file using batch operations.
            Ensure the WHOLE PATH exists. Folders in the db_filename will not be created!
            If an open `db` connection is supplied it is reused (and left open) so a caller
            can save many conversations over a single connection; otherwise one is opened and closed here.
        """
        my_db = db
        owns_db:bool = db is None
        try:
            if my_db is None:
                my_db = AIConversationDB(db_filename)

            # Build rows list for batch upsert
            rows = []
            for msg_no, message in enumerate(self.get_messages()):
                url = message.get_file_url() if message.get_type() == rc.AIC_TYPE_FILE else message.get_url()
                rows.append((msg_no, message.get_role(), message.get_type(), message.get_text(), url, message.is_sticky(), message.get_model(), message.get_file_name()))

            # Single-transaction batch: delete old + insert all new
            my_db.upsert_rows_batch(chat_name, chat_model, rows)

        except FileNotFoundError:
            print(f"Data NOT Saved to DB. Database file not found: {db_filename}")
            print("Check the property file to change current db_filename settings. Note the WHOLE path need to exists (folders will not be created)!")
            raise

        except Exception as e:
            print(f"Conversation Data NOT Saved to DB. An error occurred: {e}")
            raise

        finally:
            if owns_db and my_db:
                my_db.close()

        return


    def delete_from_db(self, db_filename:str, chat_name:str)->None:
        """ Delete the conversation from a SQLite database file.
        """
        my_db = None
        try:
            my_db = AIConversationDB(db_filename)
            my_db.delete_chat(chat_name)
        finally:
            if my_db:
                my_db.close()
        return
    


if __name__ == '__main__':
    print("AIConversation.py - not an executable module.")
