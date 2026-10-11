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
ROLE_TOOL        = "tool"        #what a tool returned, recorded by the caller after running it (see AIC_TYPE_TOOL_RESULT)
ROLE_INTERNAL    = "internal"    #message with ROLE INTERNAL will never be sent to AI model and will not count in any sizing, history size etc....

#conversation constants
#types of messages
AIC_TYPE_TEXT        = "text"
AIC_TYPE_IMAGE_URL   = "image_url"
AIC_TYPE_FILE        = "file"
AIC_TYPE_INTERNAL    = "internal" #used to send "commands" to the system
AIC_TYPE_REASONING   = "reasoning" #used to store reasoning steps (when part of the model output). Not sent to the model as part of the conversation memory
AIC_TYPE_WARNING     = "warning"   #application warning, recorded under ROLE_INTERNAL
AIC_TYPE_NEWTOPIC    = "newtopic"  #explicit topic boundary, recorded under ROLE_INTERNAL
AIC_TYPE_AGENT       = "agent"     #recorded agent output under ROLE_ASSISTANT, never sent in memory
AIC_TYPE_TOOL_CALL   = "tool_call"   #the assistant asked for one or more tools to be run. RECORD ONLY: kept in the history, never sent back as part of memory
AIC_TYPE_TOOL_RESULT = "tool_result" #what one tool returned, answering one tool_call_id. RECORD ONLY, like AIC_TYPE_TOOL_CALL

#The (role, type) contract, in one place. This table IS the spec: the AIMessage constructor
#validates every message against it and quotes it back in its error messages, so there is nothing
#else to keep in sync. Read it as "which content types may each role carry":
#  - system / developer : text only (a system prompt is text)
#  - user               : text, plus image or file attachments
#  - assistant          : text and reasoning (model output), images/files it may return, and
#                         tool calls (a request to run tools, also model output), and agent records
#  - tool               : a tool result only - the one thing a tool ever says
#  - internal           : commands, warnings and topic boundaries - never attachments
#A role absent from this table is not a valid message role.
AIC_ALLOWED_TYPES_BY_ROLE = {
    ROLE_SYSTEM:    {AIC_TYPE_TEXT},
    ROLE_DEVELOPER: {AIC_TYPE_TEXT},
    ROLE_USER:      {AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE},
    ROLE_ASSISTANT: {AIC_TYPE_TEXT, AIC_TYPE_IMAGE_URL, AIC_TYPE_FILE, AIC_TYPE_REASONING, AIC_TYPE_TOOL_CALL, AIC_TYPE_AGENT},
    ROLE_TOOL:      {AIC_TYPE_TOOL_RESULT},
    ROLE_INTERNAL:  {AIC_TYPE_INTERNAL, AIC_TYPE_WARNING, AIC_TYPE_NEWTOPIC},
}

#the content types that are RECORDED but never SENT: they live in the history, cost 0 tokens
#(the token estimate is "what will the next request cost", and these are never in it) and are
#left out of memory by get_memory_messages(). INTERNAL is here unconditionally; REASONING can be
#let back in with add_reasoning=True; warnings, topic markers, agent records and the two TOOL types
#cannot be let back in at all, see
#AIConversation.__may_enter_memory() for why.
AIC_TOOL_TYPES = {AIC_TYPE_TOOL_CALL, AIC_TYPE_TOOL_RESULT}

#the text of a topic marker: get_memory_messages() stops at the most recent AIC_TYPE_NEWTOPIC
#or legacy internal message with this text.
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
