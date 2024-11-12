from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, config_list_from_json

# Load LLM configuration
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

# Initialize Agents
user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

jr_agent = AssistantAgent(
    name="JuniorAgent",
    llm_config=llm_config,
    system_message=(
        "You are a junior assistant. Answer questions to the best of your ability. "
        "Reply TERMINATE when done."
    ),
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

sr_agent = AssistantAgent(
    name="SeniorAgent",
    llm_config=llm_config,
    system_message=(
        "You are a senior assistant. Verify and improve answers provided by junior assistants. "
        "Reply TERMINATE when done."
    ),
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

qa_agent = AssistantAgent(
    name="QAAgent",
    llm_config=llm_config,
    system_message=(
        "You are a QA agent. Ensure the accuracy and correctness of responses. "
        "Reply TERMINATE when done."
    ),
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Define FSM transitions
fsm_transitions = {
    user: [jr_agent],
    jr_agent: [sr_agent],
    sr_agent: [qa_agent],
    qa_agent: [],  # No further transitions after QAAgent
}

# Create the GroupChat with FSM transitions
agents = [user, jr_agent, sr_agent, qa_agent]

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions=fsm_transitions,
    speaker_transitions_type="allowed",
)

# Create the GroupChatManager
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Define the task
task = "What is the capital of France?"

# Start the group chat using the manager
user.initiate_chat(manager, message=task)

# Print the conversation
print("\nGroup Chat Conversation:")
for msg in group_chat.messages:
    print(f"{msg.get('role', 'Unknown')}: {msg['content']}\n")
