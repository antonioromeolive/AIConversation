"""
rom_const.py - Central Configuration Constants

This module defines all constants used throughout the AIConversation project.
Organized by functional areas:
- Message roles and types

Import as: import rom_const as rc
"""
#message roles
ROLE_USER        = "user"
ROLE_SYSTEM      = "system"
ROLE_ASSISTANT   = "assistant"
ROLE_DEVELOPER   = "developer"
ROLE_INTERNAL    = "internal"    #message with ROLE INTERNAL will never be sent to AI model and will not count in any sizing, history size etc....

#conversation constants
#types of messages
AIC_TYPE_TEXT        = "text"
AIC_TYPE_IMAGE_URL   = "image_url"
AIC_TYPE_FILE        = "file"
AIC_TYPE_INTERNAL    = "internal" #used to send "commands" to the system
AIC_TYPE_REASONING   = "reasoning" #used to store reasoning steps (when part of the model output). Not sent to the model as part of the conversation memory

#The (role, type) contract, in one place. This table IS the spec: the AIMessage constructor
#validates every message against it and quotes it back in its error messages, so there is nothing
#else to keep in sync. Read it as "which content types may each role carry":
#  - system / developer : text only (a system prompt is text)
#  - user               : text, plus image or file attachments
#  - assistant          : text and reasoning (model output), and images/files it may return
#  - internal           : its command text only - never an image, never conversation content
#A role absent from this table is not a valid message role.
AIC_ALLOWED_TYPES_BY_ROLE = {
    ROLE_SYSTEM:    {AIC_TYPE_TEXT},
    ROLE_DEVELOPER: {AIC_TYPE_TEXT},
    ROLE_USER:      {AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE},
    ROLE_ASSISTANT: {AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE, AIC_TYPE_REASONING},
    ROLE_INTERNAL:  {AIC_TYPE_INTERNAL},
}

#the ONE internal marker this class interprets: get_memory_messages() stops at the most recent one.
#Any other text passed to add_comment() ("/MODEL:gpt-4o", "user changed provider", ...) is an
#application-defined note: stored verbatim in the history, never parsed, never sent to a model.
AIC_COMMAND_NEWTOPIC         = "/NEWTOPIC"

#substituted when the caller strips a content type it knows the target model cannot accept.
#the stored conversation keeps the original content: only the copy handed to the caller is degraded.
AIC_PLACEHOLDER_IMAGE   = "[image omitted: not supported by the target model]"
AIC_PLACEHOLDER_FILE    = "[file omitted: not supported by the target model]"
AIC_PLACEHOLDER_GENERIC = "[attachment omitted: not supported by the target model]"

#which placeholder stands in for which stripped content type
AIC_PLACEHOLDER_BY_TYPE = {
    AIC_TYPE_IMAGE_URL: AIC_PLACEHOLDER_IMAGE,
    AIC_TYPE_FILE:      AIC_PLACEHOLDER_FILE
}

#message status
AIC_DEFAULT_SYSTEM_PROMPT       = "You are an AI assistant trying to be useful"
AIC_DEFAULT_MEMORY_SIZE         = 6
AIC_MODEL_NAME_FOR_TOKEN_COUNT  = "gpt-4o"


if __name__ == '__main__':
    print("rom_const.py - not an executable module.")
