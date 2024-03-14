class Command:
    # COMMANDS from USER to CORE-AI
    NEW_TALE = "new_tale"
    SPIN_OFF = "spin_off"
    UPDATE_PROFILE = "update_profile"
    NEW_PROFILE = "new_profile"
    CONVERSATION_RECOVERY = "conversation_recovery"
    LINK_USER_WITH_CONVERSATIONS = "link_user_with_conversations"
    USER_MESSAGE = "user_message"

    # COMMANDS from ORCH to USER
    # ACKNOWLEDGEMENT: REPEAT USER COMMAND and add ACK flag
    # ACKNOWLEDGEMENT = "ack" (not a real command)

    # COMMANDS from CHAT to USER
    PROGRESS_UPDATE = "progress_update"
    MESSAGE_FOR_HUMAN = "message_for_human"
    CHAT_COMPLETED = "done"
    STATUS_UPDATE = "status_update"
