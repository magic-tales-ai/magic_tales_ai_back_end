from services.openai_assistants.persona_rag.agents.group import AgentGroup
from services.openai_assistants.user_profile_agent.user_profile_agent import UserProfileAgent
from services.openai_assistants.contextual_retrieval_agent.contextual_retrieval_agent import ContextualRetrievalAgent
from services.openai_assistants.live_session_agent.live_session_agent import LiveSessionAgent
from services.openai_assistants.document_ranking_agent.document_ranking_agent import DocumentRankingAgent
from services.openai_assistants.feedback_agent.feedback_agent import FeedbackAgent
from services.openai_assistants.chat_assistant.chat_assistant import ChatAssistant
from services.openai_assistants.supervisor_assistant.supervisor_assistant import SupervisorAssistant



class MagicTalesAgentGroup(AgentGroup):
    def __init__(self, agent_dic={}):
        super().__init__(agent_dic)

    def initialize_agents(self, config):        
        # Initialize the Helper Assistant
        self.add_agent(ChatAssistant(config.chat_assistant), "chat_assistant")
        self.add_agent(SupervisorAssistant(config.supervisor_assistant), "supervisor_agent")
        self.add_agent(UserProfileAgent(config.user_profile_agent), "user_profile_agent")
        self.add_agent(ContextualRetrievalAgent(config.contextual_retrieval_agent), "contextual_retrieval_agent")
        self.add_agent(LiveSessionAgent(config.live_session_agent), "live_session_agent")
        self.add_agent(DocumentRankingAgent(config.document_ranking_agent), "document_ranking_agent")
        self.add_agent(FeedbackAgent(config.feedback_agent), "feedback_agent")