from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
llm_config = {"config_list": config_list}

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Initialize Agents
user_proxy = UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

jr_agent = AssistantAgent(
    name="JuniorAgent",
    system_message="You are a junior assistant. Answer questions to the best of your ability.",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

sr_agent = AssistantAgent(
    name="SeniorAgent",
    llm_config=llm_config,
    system_message="You are a senior assistant. Verify and improve answers provided by junior assistants. Reply TERMINATE when done",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

qa_agent = AssistantAgent(
    name="QAAgent",
    llm_config=llm_config,
    system_message="You are a QA agent. Ensure the accuracy and correctness of responses.",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
)

# Create Group Chats
groupchat_1 = GroupChat(agents=[user_proxy, jr_agent, sr_agent], messages=[], max_round=10)
groupchat_2 = GroupChat(agents=[user_proxy, sr_agent, qa_agent], messages=[], max_round=10)

# Create Group Chat Managers
manager_1 = GroupChatManager(
    groupchat=groupchat_1,
    name="GroupChatManager1",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

manager_2 = GroupChatManager(
    groupchat=groupchat_2,
    name="GroupChatManager2",
    llm_config=llm_config,  
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

# Initialize User Agent
user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

# Define the task
task = "What is the capital of France?"

# Initiate Chats
user.initiate_chats(
    [
        # First nested group chat: User_proxy, JuniorAgent, SeniorAgent
        {"recipient": manager_1, "message": task, "summary_method": "last_msg"},
        # Second nested group chat: User_proxy, SeniorAgent, QAAgent
        {"recipient": manager_2, "message": task, "summary_method": "last_msg"},
    ]
)

# Print the conversations
print("\nGroup Chat 1 Conversation:")
for msg in groupchat_1.messages:

    print(f"{msg['name']}: {msg['content']}\n")

print("\nGroup Chat 2 Conversation:")
for msg in groupchat_2.messages:
    print(f"{msg['name']}: {msg['content']}\n")
