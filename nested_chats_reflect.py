from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Load LLM configuration
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

# Initialize Agents
user = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

jr_agent = AssistantAgent(
    name="JuniorAgent",
    llm_config=llm_config,
    system_message="You are a junior assistant. Answer questions to the best of your ability. Reply TERMINATE when done, or if the query appears redundant, or is stating that an assistant is ready to help.",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

sr_agent = AssistantAgent(
    name="SeniorAgent",
    llm_config=llm_config,
    system_message="You are a senior assistant. Verify and improve answers provided by junior assistants. Reply TERMINATE when done providing your improved response.",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

qa_agent = AssistantAgent(
    name="QAAgent",
    llm_config=llm_config,
    system_message="You are a QA agent. Ensure the accuracy and correctness of responses. Reply TERMINATE when verification is completed.",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Define Nested Chat Message Functions
def sr_agent_message(recipient, messages, sender, config):
    # Get the last message from JuniorAgent to User
    last_message = jr_agent.chat_messages_for_summary(user)[-1]['content']
    return f"Please verify and improve the following answer from JuniorAgent, end with TERMINATE:\n\n{last_message}"

def qa_agent_message(recipient, messages, sender, config):
    # Get the last message from SeniorAgent to User
    last_message = jr_agent.chat_messages_for_summary(user)[-1]['content']

    return f"Please check the accuracy of the following answer verified by SeniorAgent, end with TERMINATE:\n\n{last_message}"

# Register Nested Chats
user.register_nested_chats(
    [
        {
            "recipient": sr_agent,
            "message": sr_agent_message,
            "summary_method": "last_msg",
            "max_turns": 2
        },
        {
            "recipient": qa_agent,
            "message": qa_agent_message,
            "summary_method": "last_msg",
            "max_turns": 2
        }
    ],
    trigger=jr_agent  # Nested chats are triggered when JuniorAgent sends a message
)

# Define the task
task = "What is the capital of France?"

# Initiate Chat with JuniorAgent
res = user.initiate_chat(
    recipient=jr_agent,
    message=task,
    max_turns=2,
    summary_method="last_msg"
)

print(res)
# Print the conversations
