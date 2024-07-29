from services.openai_assistants.persona_rag.workflows.workflow import Workflow
from services.openai_assistants.magic_tales_global_memory import GlobalMemory

class MagicTalesWorkflow(Workflow):
    def __init__(self, agents, global_memory: GlobalMemory, current_question="", current_passages=[]):
        super().__init__(agents, workflow_list=[], global_memory=global_memory.to_dict(), 
                         current_question=current_question, current_passages=current_passages)
        self.global_memory = global_memory

    def profile_creation_flow(self):
        # Implement logic for profile creation/selection
        user_profile_agent = self.agents.agent_dic["user_profile_agent"]
        chat_assistant = self.agents.agent_dic["chat_assistant"]

        # Check if user has existing profiles
        existing_profiles = self.global_memory.get("existing_profiles", [])

        if not existing_profiles:
            # Suggest creating a new profile
            chat_assistant.send_message("It looks like you don't have any profiles yet. Let's create one!")
            new_profile = user_profile_agent.create_new_profile()
            self.global_memory.update("current_profile", new_profile)
        else:
            # Ask user to select a profile or create a new one
            chat_assistant.send_message("You have existing profiles. Would you like to use one of them or create a new one?")
            # Implement logic to handle user's choice

    def story_feature_gathering_flow(self):
        # Implement logic for gathering story features
        chat_assistant = self.agents.agent_dic["chat_assistant"]
        contextual_retrieval_agent = self.agents.agent_dic["contextual_retrieval_agent"]

        # List of story features to gather
        features = ["genre", "language", "length", "characters", "plot", "setting", "theme"]

        story_features = {}
        for feature in features:
            # Ask user about each feature
            chat_assistant.send_message(f"Let's talk about the {feature} for your story.")
            user_response = chat_assistant.get_user_response()
            
            # Use contextual retrieval to enhance the response
            enhanced_response = contextual_retrieval_agent.retrieve_context(user_response)
            
            story_features[feature] = enhanced_response

        self.global_memory.update("story_features", story_features)

    def synopsis_generation_flow(self):
        # Implement logic for synopsis generation and approval
        chat_assistant = self.agents.agent_dic["chat_assistant"]
        cognitive_agent = self.agents.agent_dic["cognitive_agent"]

        story_features = self.global_memory.get("story_features", {})
        current_profile = self.global_memory.get("current_profile", {})

        # Generate initial synopsis
        initial_synopsis = chat_assistant.generate_synopsis(story_features, current_profile)

        # Refine synopsis using cognitive agent
        refined_synopsis = cognitive_agent.refine_response(initial_synopsis, current_profile, story_features)

        # Present synopsis to user and get approval
        approved = False
        while not approved:
            chat_assistant.send_message(f"Here's the synopsis for your story:\n\n{refined_synopsis}\n\nDo you approve this synopsis?")
            user_response = chat_assistant.get_user_response()
            if user_response.lower() in ['yes', 'approve', 'good']:
                approved = True
            else:
                # Get feedback and refine synopsis
                chat_assistant.send_message("What would you like to change in the synopsis?")
                feedback = chat_assistant.get_user_response()
                refined_synopsis = cognitive_agent.refine_response(refined_synopsis, current_profile, story_features, feedback)

        self.global_memory.update("approved_synopsis", refined_synopsis)

    def execute(self):
        super().execute()
        self.check_supervisor_intervention()

    def check_supervisor_intervention(self):
        supervisor_agent = self.agents.agent_dic["supervisor_agent"]
        chat_assistant = self.agents.agent_dic["chat_assistant"]

        # Check for intervention after each chat assistant response
        latest_chat_response = chat_assistant.get_latest_response()
        intervention_needed = supervisor_agent.check_for_intervention(latest_chat_response)

        if intervention_needed:
            intervention_message = supervisor_agent.generate_intervention()
            chat_assistant.send_message(intervention_message)
            # Handle the intervention (e.g., correcting information, clarifying points)