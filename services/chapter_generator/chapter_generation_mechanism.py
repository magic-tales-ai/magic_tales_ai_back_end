import os
import copy
import json
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import messages_to_dict
from omegaconf import DictConfig, OmegaConf

from .chapter_generator_LLM import (
    ChapterGeneratorLLM,
)
from .chapter_critic_LLM import (
    ChapterCriticLLM,
)

from .utils.saving_loading_utils import (
    create_top_level_chapter_folder,
    save_all_chapters_data_to_csv,
    save_content_to_creation_data_storage_folder,
    save_best_chapter_to_output_file,
    save_chains_of_thoughts,
    initialize_chapter_info_dict,
    load_root_from_content_file,
    save_general_creation_info,
    save_tree_state,
    load_tree_state,
)

from .utils.viz_tree_utils import animate_mcts_evolution
from .utils.chapter_tree import ChapterTree
from .utils.chapter_node import ChapterNode
from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ChapterGenerationMechanism:
    """
    The main class for Magic Tales with LLM Chapter Generation Mechanism, using MCTS (Monte Carlo Tree Search algorithm).
    To decide on a chapter, we generate a chapter and create a critique for next round, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we reach a leaf node.
    """

    def __init__(
        self,
        config: DictConfig,
        story_blueprint: Dict[str, str],
        previous_chapter_content: str,
    ):
        # Set the same seed for
        random.seed(config.mcts.seed_value)

        self.config = config

        # Initialize the with ChapterNode.scoring_config with the weights from the config to score chapters
        # ChapterNode.set_chapter_scoring_config(config.chapter_scoring)

        self.chapter_number: int = 1
        self.total_number_chapters: int = 1

        # Initialize the Global & static node counter to 0
        ChapterNode.node_counter = 0

        main_llm = ChatOpenAI(
            model_name=self.config.main_llm.model,
            temperature=self.config.main_llm.temperature,
            # max_tokens=self.config.main_llm.max_tokens,
            # model_kwargs={"top_p": self.config.main_llm.top_p},
            verbose=True,
            request_timeout=self.config.main_llm.request_timeout,
        )

        parser_llm = ChatOpenAI(
            model_name=self.config.parser_llm.model,
            temperature=self.config.parser_llm.temperature,
            # max_tokens=self.config.parser_llm.max_tokens,
            # model_kwargs={"top_p": self.config.parser_llm.top_p},
            verbose=True,
            request_timeout=self.config.parser_llm.request_timeout,
        )

        # Initialize Generation and Critic+Scorer Agents
        self.chapter_generator = ChapterGeneratorLLM(
            main_llm=main_llm,
            parser_llm=parser_llm,
            story_blueprint=story_blueprint,
            previous_chapter_content=previous_chapter_content,
            num_outputs=self.config.main_llm.num_responses,
        )
        self.chapter_critic = ChapterCriticLLM(
            main_llm=main_llm,
            parser_llm=parser_llm,
            story_blueprint=story_blueprint,
            previous_chapter_content=previous_chapter_content,
            num_outputs=self.config.parser_llm.num_responses,
        )

        self.top_level_creation_folder = None
        self.subfolders = {
            "results": "results",
            "content": "content",
            "best_chapter": "best_chapter",
            "chain_of_thoughts_conversations": "chain_of_thoughts_conversations",
            "tree_evolution": "tree_evolution",
        }

        self.current_creation_cycle = 0
        self.chapter_tree = ChapterTree(self.config.mcts, self.config.chapter_scoring)

    @staticmethod
    def build_search_path(node: ChapterNode) -> List["ChapterNode"]:
        """
        Builds a search path from the given node back to the root.

        :param node: The node from which to start building the path.
        :return: A list containing nodes from the given node to the root.
        """
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse the list to start from the root

    def create_chapter(
        self,
        chapters_folder: str,
        chapter_number: int,
        total_number_chapters: int,
    ) -> Optional[Dict[str, Any]]:
        """
        This method facilitates the creation process of a new chapter via a Monte Carlo Tree Search (MCTS).

        The process involves the following steps:

        - Initialize/reset the best chapter, maximum tree depth, chapter rows, and results episode rows.
        - Check if it should continue from the last creation session or start a new creation session.
            - If it continues from the last session, it loads the tree state and fetches the best chapter.
            - If it's a new session, it creates a new folder for saving creation data and determines the root node of the tree.
        - Initialize the MinMaxStats object for tracking the minimum and maximum statistics during the creation process.
        - Start the MCTS for a specified number of creation cycles. For each cycle:
            - Select a leaf node from the tree that balances exploration and exploitation.
            - Generate a new chapter if the leaf node doesn't have any chapter content associated with it.
            - Critique the chapter content.
            - Gather all chapter related data: content, critique, etc.
            - Expand the tree with a new node and backpropagate the chapter "quality score".(look at ChapterNode class for info: Calculate the weighted score of the node.)
            - Check for low performance and generate fresh chapters if necessary.
            - Save the tree state at each iteration and check the success ratio of the best chapter. If it meets the success ratio threshold, break the loop.
        - Handle any exceptions that occur during the creation process and log the error message.
        - Finally, save the tree state, all episodes data, all chapters data, and general creation info to CSV files.

        Returns:
            dict or None: A dictionary containing information about the best chapter.
            The dictionary contains the following keys:
                "chapter_generator_response_dict": A dictionary containing information about the chapter.
                And other keys related to the chapter's critique and performance metrics.
        """
        try:
            # Initialize the chapter Acquisition (or creation) Session
            self._initialize_chapter_creation_session(
                chapters_folder, chapter_number, total_number_chapters
            )

            chapters_rows = []

            for i in range(1, self.config.mcts.max_num_creation_cycles + 1):
                logger.info(
                    f"creation cycle {i} out of {self.config.mcts.max_num_creation_cycles}"
                )
                self.current_creation_cycle = i

                # Node Selection
                logger.info(
                    "Selecting a leaf node from the tree while balancing exploitation and exploration"
                )

                if self.config.mcts.use_greedy_node_selection and isinstance(
                    self.chapter_tree.best_chapter, ChapterNode
                ):
                    node = self.chapter_tree.best_chapter
                    search_path = self.build_search_path(node)
                    # Force create a new leaf node from the best node, to be used for improvements
                    node = self.chapter_tree.expand_tree(node)
                    search_path.append(node)
                else:
                    node = self.chapter_tree.root
                    search_path = [node]
                    while node.expanded():
                        node = self.chapter_tree.select_child(node)
                        search_path.append(node)

                # chapter content Generation
                if node.is_empty():
                    new_chapter_info_dict = self._generate_chapter(
                        chapter_to_improve=node.parent
                    )
                    if new_chapter_info_dict["chapter_generator_success"]:
                        node.set_chapter_dict(new_chapter_info_dict)
                    else:
                        logger.info(
                            "Failed to generate a chapter content. We'll skip this round and continue"
                        )
                        continue

                # chapter Critique
                _, chapter_info_dict_with_crtique = self._generate_critique(
                    node.chapter_info_dict
                )
                node.set_chapter_dict(chapter_info_dict_with_crtique)

                # Used solely for logging statistics; 'chapter_rows' doesn't influence content generation.
                self._add_chapter_data_to_chapter_rows(node, chapters_rows)

                # Tree Expansion
                logger.info("Expanding the tree with 1 node (leaf)")
                self.chapter_tree.expand_tree(node)

                # Back-propagation
                self.chapter_tree.backpropagate(search_path)

                # Check for low performance and generate fresh chapters if needed
                self.chapter_tree.check_performance_and_generate_fresh_chapters()

                # If we want to generate a Visulization of the Tree evolving
                if self.config.output_artifacts.generate_tree_evolution_viz:
                    save_tree_state(
                        chapter_generation_mechanism=self, save_tree_evolution=True
                    )

                # Save the latest tree state for recovery or continue creation
                save_tree_state(chapter_generation_mechanism=self)

                logger.info(f"Best chapter so far is {self.chapter_tree.best_chapter}")

                # If the best chapter has surpased threshold we stop the process
                if self.chapter_tree.success():
                    logger.info(
                        f"Best chapter has surpased the success threshold! We are done"
                    )
                    break

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                f"An error occurred on the creation cycle.\nException: {e}\n{tb}"
            )
            raise e

        finally:
            # No matter what happens we try to save everything
            save_tree_state(chapter_generation_mechanism=self)
            save_all_chapters_data_to_csv(
                top_level_creation_folder=self.top_level_creation_folder,
                results_subfolder=self.subfolders.get("results", ""),
                chapters_rows=chapters_rows,
            )
            save_general_creation_info(chapter_generation_mechanism=self)
            save_chains_of_thoughts(
                top_level_creation_folder=self.top_level_creation_folder,
                chain_of_thoughts_conversations_subfolder=self.subfolders.get(
                    "chain_of_thoughts_conversations", ""
                ),
                outputs_config=self.config.output_artifacts,
                root=self.chapter_tree.root,
            )

        # Generate a Visulization of the Tree evolving if configured to do so
        if self.config.output_artifacts.generate_tree_evolution_viz:
            animate_mcts_evolution(self.top_level_creation_folder)

        return self._finalize_chapter_acquisition_session()

    def _initialize_chapter_creation_session(
        self,
        chapters_folder: str,
        chapter_number: int,
        total_number_chapters: int,
    ) -> None:
        """
        Initialize the chapter acquisition session.

        Initialize/reset the best chapter, maximum tree depth, chapter rows, and results episode rows.
        Check if it should continue from the last creation session or start a new creation session.
            - If it continues from the last session, it loads the tree state and fetches the best chapter.
            - If it's a new session, it creates a new folder for saving creation data and determines the root node of the tree.
        """
        self.chapter_tree.reset()
        self.chapter_number = chapter_number
        self.total_number_chapters = total_number_chapters
        self.chapters_folder = chapters_folder

        if self.config.mcts.continue_from_last_creation_session and load_tree_state(
            chapter_generation_mechanism=self,
            top_level_creation_folder=self.chapters_folder,
        ):
            logger.info(
                "Tree succesfully loaded from previous session. Continuing last creation session"
            )
        else:
            self._start_new_chapter_creation_session()

    def _start_new_chapter_creation_session(self) -> None:
        """
        Start a new chapter creation session. This involves creating a new top-level creation folder
        and deciding whether to run with an initial content seed/example or from a blank chapter tree.
        """
        logger.info("creation a new chapter")
        self._create_top_level_creation_folder()

        if self.config.mcts.override_root_with:
            self._run_chapter_creation_with_initial_content_example()
        else:
            self._run_chapter_creation_with_no_initial_content_example()

        # The Root is not a normal node and we don't want to go through a creation cycle on it, therefore we must create at least 1 child
        self.chapter_tree.expand_root()

    def _create_top_level_creation_folder(self) -> None:
        """
        Create a new top level creation folder to save all data.
        """
        self.top_level_creation_folder = create_top_level_chapter_folder(
            chapters_folder_data_storage=self.chapters_folder,
            chapter_number=self.chapter_number,
            subfolders=self.subfolders,
            name_for_this_creation_run=self.config.mcts.name_for_this_creation_run,
        )

        if self.top_level_creation_folder is None:
            logger.error("Failed to create a top level folder to save creation data.")
            raise RuntimeError(
                "Failed to create a top level folder to save creation data. We cannot continue"
            )

        self.best_chapter_output_file = os.path.join(
            self.top_level_creation_folder, "best_chapter.txt"
        )

    def _run_chapter_creation_with_initial_content_example(self) -> None:
        """
        Run chapter creation with an initial content example.
        This mode attempts to load the root from a JSON file containing a pre-written content snippet,
        which is used to seed the root of the chapter tree and guide the initial steps of the chapter generation process.
        """
        logger.info(
            "Running chapter creation with the initial content example provided"
        )
        self.chapter_tree.root = load_root_from_content_file(
            self.config.mcts.override_root_with, self.cofig.chapter_scoring
        )

        if self.chapter_tree.root is None:
            logger.info(
                "Error loading root from JSON, falling back to chapter creation without any content example provided"
            )
            self.chapter_tree.root = ChapterNode(
                scoring_config=self.config.chapter_scoring
            )
        else:
            logger.info("Root with example content loaded!")
            content = self.chapter_tree.root.chapter_info_dict.get(
                "chapter_generator_response_dict", {}
            ).get("content", " ")
            logger.info(f"chapter loaded:\n{content}")

    def _run_chapter_creation_with_no_initial_content_example(self) -> None:
        """
        Run chapter creation without any content example provided.
        This mode starts the chapter creation process without any content example,
        allowing the system to explore and generate chapters autonomously without pre-defined guidance.
        """
        logger.info("Running chapter creation without any content example provided")
        self.chapter_tree.root = ChapterNode(scoring_config=self.config.chapter_scoring)

    def _generate_chapter(self, chapter_to_improve: ChapterNode) -> Dict[str, str]:
        """
        Generate a new chapter based on the provided chapter information dictionary that needs to be improved (using the critique and prev content).

        :param chapter_to_improve: A node containing information about the chapter that needs to be improved.
        :return: A dictionary containing information about the generated chapter.
        """
        logger.info("Generating chapter content using a LLM")
        chapter_to_improve_dict = initialize_chapter_info_dict(
            chapter_number=self.chapter_number,
            total_num_chapters=self.total_number_chapters,
        )

        if (
            chapter_to_improve is None
            or chapter_to_improve.chapter_info_dict is {}
            or not chapter_to_improve.chapter_info_dict.get(
                "chapter_generator_response_dict"
            )
        ):
            logger.info("Generating chapter from scratch")
        else:
            logger.info(
                f"Generating a new chapter to improve node {chapter_to_improve.node_id}"
            )
            chapter_to_improve_dict = copy.deepcopy(
                chapter_to_improve.chapter_info_dict
            )

        chapter_generator_success = False
        chapter_candidate: Dict[str, str] = {}
        chapter_generator_prompt_messages = ["", ""]

        for attempt in range(1, self.config.main_llm.max_retries + 1):
            try:
                (
                    chapter_candidates,
                    chapter_generator_prompt_messages,
                ) = self.chapter_generator.generate_chapter_content(
                    chapter_to_improve_dict
                )

                # We might or might not have failed to generate a new chapter. After every use of this function you should check for success
                chapter_generator_success, chapter_candidate = chapter_candidates[0]

                if not chapter_generator_success:
                    raise RuntimeError("No chapter candidates generated")

                break

            except Exception as e:
                chapter_candidate["content"] = None
                chapter_generator_success = False
                tb = traceback.format_exc()
                logger.info(
                    f"We couldn't create the chapter (content). Attempt {attempt} of {self.config.main_llm.max_retries}.\nException: {e}\n{tb}"
                )

        # Here we create a New chapter dict that will be held on the node
        new_chapter_info_dict = initialize_chapter_info_dict(
            chapter_generator_success=chapter_generator_success,
            chapter_generator_prompt_messages=chapter_generator_prompt_messages,
            chapter_generator_response_dict=copy.deepcopy(chapter_candidate),
        )
        self._check_visualize_chapter_generator_response(new_chapter_info_dict)
        return new_chapter_info_dict

    def _check_visualize_chapter_generator_response(
        self, chapter_info_dict: Dict[str, str]
    ) -> None:
        """
        Checks the configuration and visualizes the response from the chapter generator.

        This method prints the rationale, plan, etc associated with a chapter.
        The output is controlled by the `visualize_chapter_generator_response` configuration option.

        :param chapter_info_dict: A dictionary containing information about the chapter, including the response from the chapter generator.
        """
        if (
            self.config.output_artifacts.visualize_chapter_generator_response
            and chapter_info_dict.get("chapter_generator_success", False)
        ):
            # rationale = chapter_info_dict["chapter_generator_response_dict"][
            #     "rationale"
            # ]
            plan = chapter_info_dict["chapter_generator_response_dict"]["plan"]
            # synopsis = chapter_info_dict["chapter_generator_response_dict"]["synopsis"]
            chapter_content = chapter_info_dict["chapter_generator_response_dict"][
                "content"
            ]
            logger.info(f"\033[34mChapter:\033[0m")
            # logger.info(f"\033[34m  Rationale:\033[0m")
            # logger.info(f"\033[34m{rationale}\033[0m")
            logger.info(f"\033[34m  Plan:\033[0m")
            logger.info(f"\033[34m{plan}\033[0m")
            # logger.info(f"\033[34m  Synopsis:\033[0m")
            # logger.info(f"\033[34m{synopsis}\033[0m")
            logger.info(f"\033[34m  content:\033[0m")
            logger.info(f"\033[34m{chapter_content}\033[0m")

    def _generate_critique(
        self, chapter_info_dict: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Generate a critique based on the provided chapter information dictionary, specifically using the content and the metrics on the dict.

        :param chapter_info_dict: A dictionary containing information about the chapter that needs to be critiqued.
        :return: Tuple with success status and updated chapter information dictionary
        """
        logger.info("Generating a critique using a LLM")

        for attempt in range(1, self.config.parser_llm.max_retries + 1):
            try:
                (
                    chapter_critics,
                    chapter_critic_prompt_messages,
                ) = self.chapter_critic.generate_critique(chapter_info_dict)

                if chapter_critics is None or not chapter_critics:
                    raise RuntimeError("No chapter criticism generated")

                chapter_critic_success, chapter_critic_response_dict = chapter_critics[
                    0
                ]

                chapter_info_dict["chapter_critic_success"] = chapter_critic_success
                chapter_info_dict["chapter_critic_prompt_messages"] = (
                    chapter_critic_prompt_messages
                )
                chapter_info_dict["chapter_critic_response_dict"] = (
                    chapter_critic_response_dict
                )

                self._check_visualize_chapter_critic_response(chapter_info_dict)
                return True, chapter_info_dict

            except RuntimeError as e:
                logger.info(
                    f"Couldn't create the critique for the provided chapter content. Attempt {attempt} of {self.config.parser_llm.max_retries}. Error: {str(e)}"
                )
        else:
            logger.error("Failed to generate a critique after all attempts")
            return False, chapter_info_dict

    def _check_visualize_chapter_critic_response(
        self, chapter_info_dict: Dict[str, str]
    ) -> None:
        """
        Checks the configuration and visualizes the response from the chapter critic.

        This method prints the critique, rationale, and score associated with a chapter.
        The output is controlled by the `visualize_chapter_critic_response` configuration option.

        :param chapter_info_dict: A dictionary containing information about the chapter, including the response from the chapter critic.
        """
        if (
            self.config.output_artifacts.visualize_chapter_critic_response
            and chapter_info_dict.get("chapter_critic_success", False)
        ):
            full_ctirique = chapter_info_dict.get("chapter_critic_response_dict")
            logger.info(f"\033[34mFull Critique:\033[0m")
            logger.info(f"\033[34m{full_ctirique}\033[0m")
            # critique = chapter_info_dict["chapter_critic_response_dict"]["critique"]
            # rationale = chapter_info_dict["chapter_critic_response_dict"]["rationale"]
            # # score = chapter_info_dict["chapter_critic_response_dict"]["score_by_llm"]
            # logger.info(f"\033[34mchapter:\033[0m")
            # logger.info(f"\033[34m  Critique:\033[0m")
            # logger.info(f"\033[34m{critique}\033[0m")
            # logger.info(f"\033[34m  Rationale:\033[0m")
            # logger.info(f"\033[34m{rationale}\033[0m")
            # # logger.info(f"\033[34m  Score:\033[0m")
            # # logger.info(f"\033[34m{score}\033[0m")

    def get_value_from_dict(
        self,
        node: "ChapterNode",
        key: str,
        parent_key: Union[None, str] = None,
        default: Any = None,
    ) -> Any:
        """
        Utility function to safely retrieve values from dictionaries.

        :param node: ChapterNode object containing chapter information.
        :param key: The key to look for in the dictionary.
        :param parent_key: The parent key if the target key is nested. Default is None.
        :param default: Default value to return if key is not found. Default is None.

        :return: Value associated with the given key or the default value.
        """
        target_dict = node.chapter_info_dict
        if parent_key is not None:
            target_dict = target_dict.get(parent_key, {})
        return target_dict.get(key, default)

    def get_float_value_from_dict(
        self,
        node: "ChapterNode",
        key: str,
        parent_key: Union[None, str] = None,
        default: float = 0.0,
    ) -> float:
        """
        Utility function to safely retrieve float values from dictionaries.

        :param node: ChapterNode object containing chapter information.
        :param key: The key to look for in the dictionary.
        :param parent_key: The parent key if the target key is nested. Default is None.
        :param default: Default value to return if key is not found or not convertible to float. Default is 0.0.

        :return: Float value associated with the given key or the default value.
        """
        try:
            return float(self.get_value_from_dict(node, key, parent_key, default))
        except ValueError:
            return default

    def _gather_chapter_data(self, node: "ChapterNode") -> Dict[str, Any]:
        """
        Collects the data of a chapter generated, evaluated, and criticized by the creation process.

        :param node: Node containing chapter information.

        :return: Dictionary containing chapter data to be used for output.
        """
        metrics = [
            "plot_consistency_score",
            "character_development_score",
            "engagement_score",
            "clarity_score",
            "detailing_score",
            "language_quality_score",
            "dialogue_score",
            "emotional_impact_score",
            "originality_score",
            "closure_score",
            "alignment_with_previous_chapter_score",
            "general_alignment_with_story_score",
        ]

        chapter_data = {
            "chapter_id": node.node_id,
            "quality_score": node.get_chapter_quality_score(),
            "value_sum": node.value_sum,
            "visit_count": node.visit_count,
            "content": self.get_value_from_dict(
                node, "content", "chapter_generator_response_dict"
            ),
            "critique": self.get_value_from_dict(
                node, "critique", "chapter_critic_response_dict"
            ),
            # "rationale": self.get_value_from_dict(node, "rationale", "chapter_critic_response_dict"),
            "llm_model_name": self.chapter_generator.main_llm.model_name,
            "llm_model_temperature": self.chapter_generator.main_llm.temperature,
            "llm_additional_params": self.chapter_generator.main_llm.model_kwargs,
            "chapter_critic_prompt_messages": json.dumps(
                messages_to_dict(
                    self.get_value_from_dict(
                        node, "chapter_critic_prompt_messages", default=["", ""]
                    )
                )
            ),
            "chapter_generator_prompt_messages": json.dumps(
                messages_to_dict(
                    self.get_value_from_dict(
                        node, "chapter_generator_prompt_messages", default=["", ""]
                    )
                )
            ),
            "general_error_messages": self.get_value_from_dict(
                node, "general_error_messages"
            ),
        }

        # Add metric scores to chapter data
        for metric in metrics:
            chapter_data[metric] = self.get_float_value_from_dict(
                node, metric, "chapter_critic_response_dict"
            )

        return chapter_data

    def _add_chapter_data_to_chapter_rows(
        self, node: ChapterNode, chapters_rows: List[Any]
    ):
        """
        Collects the data of a chapter generated, evaluated and critisized by the creation process and appends it as a row in a DataFrame that collects all chapters generated.

        :param node: Node containing chapter information.
        :param chapters_rows: A list of chapter data that will build on each creation cycle.
        :return: None
        """
        logger.info(f"Gathering all the chapter data: content, critique, etc.")
        chapter_info_dict = self._gather_chapter_data(node)
        chapters_rows.append(chapter_info_dict)

    def _finalize_chapter_acquisition_session(self) -> Optional[Dict[str, Any]]:
        """
        Finalize the chapter acquisition session.

        This function checks if a best chapter was found during the chapter acquisition session.
        If a best chapter was found, its content is saved to the creation data storage folder
        and to the output file, and its information dictionary is returned.
        If no best chapter was found, the function logs an informative message and returns None.

        :return: The information dictionary of the best chapter if one was found, otherwise None.
        """

        # If no best chapter was found, or the success ratio of the best chapter is 0, we consider the chapter acquisition session to have failed
        if self.chapter_tree.best_chapter is None:
            logger.info("We have failed to acquire a chapter for this problem")
            return None
        else:
            logger.info("We have created a chapter!")

            # Extract the content of the best chapter
            best_chapter_content = self.chapter_tree.best_chapter.chapter_info_dict[
                "chapter_generator_response_dict"
            ]["content"]

            # Save the content of the best chapter to the creation data storage folder
            save_content_to_creation_data_storage_folder(
                top_level_creation_folder=self.top_level_creation_folder,
                content_subfolder=self.subfolders.get("content", ""),
                content=best_chapter_content,
                filename="best_chapter",
            )

            # Save the content of the best chapter to the output file
            save_best_chapter_to_output_file(
                content=best_chapter_content,
                full_path_and_file_name=self.best_chapter_output_file,
            )

            # Return the information dictionary of the best chapter
            return self.chapter_tree.best_chapter.chapter_info_dict
