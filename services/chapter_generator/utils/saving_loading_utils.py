import traceback
import dill
import glob
import os
from typing import TYPE_CHECKING, List, Dict, Optional, Any
from pandas import DataFrame, Series
import time
import traceback
import os
from omegaconf import DictConfig
import json
from langchain.schema import HumanMessage, SystemMessage
from typing import Callable, Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf

from services.utils.file_utils import (
    save_chain_of_thoughts_conversations_to_markdown,
    save_chain_of_thoughts_conversations_to_html,
)

from services.chapter_generator.utils.chapter_node import ChapterNode
from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)

MCTS_TREE_FILE = "mcts_state.dill"
CHAPTER_ROWS = "chapter_rows.csv"


def save_general_creation_info(
    chapter_generation_mechanism: "ChapterGenerationMechanism",
) -> None:
    """
    Save general information about the creation process to a CSV file.

    This includes the configuration used for creation, costs associated with the chapter generator and critic, total tokens used,
    successful requests, folder name for saving creation data, and the ID of the best chapter.
    """
    save_to_file = os.path.join(
        chapter_generation_mechanism.top_level_creation_folder,
        chapter_generation_mechanism.subfolders["results"],
        "general_creation_info.csv",
    )
    logger.info(
        f"Saving general information about this creation full cycle on {save_to_file}"
    )
    try:
        total_cost = (
            chapter_generation_mechanism.chapter_generator.total_cost
            + chapter_generation_mechanism.chapter_critic.total_cost
        )
        best_chapter_id = (
            chapter_generation_mechanism.chapter_tree.best_chapter.node_id
            if chapter_generation_mechanism.chapter_tree.best_chapter is not None
            else ""
        )
        creation_info_dict = {
            "config": chapter_generation_mechanism.config,
            "total_cost": total_cost,
            "chapter_generator_total_tokens": chapter_generation_mechanism.chapter_generator.total_tokens,
            "chapter_generator_prompt_tokens": chapter_generation_mechanism.chapter_generator.prompt_tokens,
            "chapter_generator_completion_tokens": chapter_generation_mechanism.chapter_generator.completion_tokens,
            "chapter_generator_total_cost: float": chapter_generation_mechanism.chapter_generator.total_cost,
            "chapter_generator_successful_requests": chapter_generation_mechanism.chapter_generator.successful_requests,
            "chapter_critic_total_tokens": chapter_generation_mechanism.chapter_critic.total_tokens,
            "chapter_critic_prompt_tokens": chapter_generation_mechanism.chapter_critic.prompt_tokens,
            "chapter_critic_completion_tokens": chapter_generation_mechanism.chapter_critic.completion_tokens,
            "chapter_critic_total_cost: float": chapter_generation_mechanism.chapter_critic.total_cost,
            "chapter_critic_successful_requests": chapter_generation_mechanism.chapter_critic.successful_requests,
            "folder_name": chapter_generation_mechanism.top_level_creation_folder,
            "best_chapter_id": best_chapter_id,
            "max_tree_depth": chapter_generation_mechanism.chapter_tree.max_tree_depth,
        }
        creation_info_series = Series(creation_info_dict)
        creation_info_df = DataFrame(creation_info_series)
        creation_info_df.to_csv(
            f"{chapter_generation_mechanism.top_level_creation_folder}/{chapter_generation_mechanism.subfolders['results']}/creation_cycle_info.csv"
        )

        if chapter_generation_mechanism.chapter_tree.best_chapter is not None:
            best_chapter_info_dict = chapter_generation_mechanism._gather_chapter_data(
                chapter_generation_mechanism.chapter_tree.best_chapter
            )
            series = Series(best_chapter_info_dict)
            best_chapter_df = DataFrame(series)
            best_chapter_df.to_csv(
                f"{chapter_generation_mechanism.top_level_creation_folder}/{chapter_generation_mechanism.subfolders['best_chapter']}/best_chapter.csv"
            )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"An error occurred while saving information about the entire creation cycle. \nException: {e}\n{tb}"
        )


def save_tree_state(
    chapter_generation_mechanism: "ChapterGenerationMechanism",
    save_tree_evolution: bool = False,
) -> None:
    """
    Save the state of the Monte Carlo Tree Search (MCTS) tree to a file.

    The state is saved as a pickle file in the top-level creation folder.
    The root node of the MCTS tree is serialized and saved.
    """
    data_to_save = {
        "root_node": chapter_generation_mechanism.chapter_tree.root,
        "node_counter": ChapterNode.node_counter,
        "scoring_config": chapter_generation_mechanism.config.chapter_scoring,
        "max_tree_depth": chapter_generation_mechanism.chapter_tree.max_tree_depth,
    }
    try:
        if save_tree_evolution:
            filename = os.path.join(
                chapter_generation_mechanism.top_level_creation_folder,
                chapter_generation_mechanism.subfolders.get("tree_evolution"),
                f"{str(chapter_generation_mechanism.current_creation_cycle)}_{MCTS_TREE_FILE}",
            )
        else:
            filename = os.path.join(
                chapter_generation_mechanism.top_level_creation_folder, MCTS_TREE_FILE
            )

        with open(filename, "wb") as f:
            dill.dump(obj=data_to_save, file=f)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Could not save tree state on {filename}.\nException: {e}\n{tb}")


def load_tree_state(
    chapter_generation_mechanism: "ChapterGenerationMechanism",
    top_level_creation_folder=None,
) -> bool:
    """
    Load the state of the MCTS tree from a file.

    If a top-level creation folder is provided, the state is loaded from there.
    If no folder is provided, the state is loaded from the most recently created folder.

    :param top_level_creation_folder: The path to the top-level creation folder from which to load the state.
    :return: True if the state was loaded successfully, False otherwise.
    """
    try:
        if top_level_creation_folder is None:
            creation_folders = glob.glob(
                f"{chapter_generation_mechanism.config.output_artifacts.chapters_folder_data_storage}/*_creation_data_*"
            )
            creation_folders.sort(key=os.path.getmtime, reverse=True)
            if creation_folders:
                top_level_creation_folder = creation_folders[0]
            else:
                logger.info("No Chapters creation folders found.")
                return False

        # Overwriting the top_level_creation_folder!!
        chapter_generation_mechanism.top_level_creation_folder = (
            top_level_creation_folder
        )
        filename = os.path.join(top_level_creation_folder, MCTS_TREE_FILE)

        if os.path.exists(filename):
            with open(filename, "rb") as f:
                saved_data = dill.load(f)
                chapter_generation_mechanism.chapter_tree.root = saved_data["root_node"]
                ChapterNode.node_counter = saved_data["node_counter"]
                chapter_generation_mechanism.chapter_tree.max_tree_depth = saved_data[
                    "max_tree_depth"
                ]
            chapter_generation_mechanism.chapter_tree.get_best_chapter()
            return True
        else:
            logger.info(f"Previous creation file not found: {filename}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.info(
            f"Failed to load the tree from '{filename}'. \nException: {e}\n{tb}"
        )
        return False


def create_top_level_chapter_folder(
    chapters_folder_data_storage: str,
    chapter_number: int,
    subfolders: dict,
    name_for_this_creation_run: Optional[str] = None,
) -> Optional[str]:
    """
    Create a top-level folder to save all data related to the current creation run.

    :param name_for_this_creation_run: The name for the current creation run. Defaults to None.
    :return: The path of the top-level creation folder, or None if there was an error creating the folder.
    """
    try:
        name_for_this_creation_run = (
            ""
            if name_for_this_creation_run is None
            else "_" + name_for_this_creation_run
        )

        # Create a directory for the creation run if it doesn't exist
        top_level_creation_folder = f"{chapters_folder_data_storage}/chapter_{chapter_number}{name_for_this_creation_run}"
        os.makedirs(top_level_creation_folder, exist_ok=True)

        # Create subfolders
        for subfolder in subfolders:
            subfolder_path = os.path.join(top_level_creation_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)

        return top_level_creation_folder

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"Error occurred while creating top level creation folder.\nException: {e}\n{tb}"
        )
        return None


def save_content_to_creation_data_storage_folder(
    top_level_creation_folder: str,
    content_subfolder: str,
    content: str,
    path: Optional[str] = None,
    filename: Optional[str] = None,
    use_timestamp: bool = False,
) -> None:
    """
    Saves a chapter content to a file on the user-specified (hidra config) creation data storage folder.

    This function checks if the directory of the file exists, and creates it if necessary.
    Then, it saves the string to the file. The file will be overwritten if it already exists.

    Parameters:
    - content (str): The content to be saved into the file.
    - path (str, optional): The directory where the file should be saved. If None, the file is saved in the current working directory. Defaults to None.
    - filename (str, optional): The name of the file. If None, the file is named 'chapter'. Defaults to None.
    - use_timestamp (bool, optional): Whether to append a timestamp to the filename. Defaults to False.

    Raises:
    - AssertionError: If the content is not a string.
    - Exception: If there was an error creating the directory or writing to the file.
    """
    assert isinstance(content, str), "content must be a string"
    timestamp = ""

    if filename is None:
        filename = "Chapter"

    if use_timestamp:
        timestamp = "_" + time.strftime("%Y%m%d-%H%M%S")

    if path is None:
        fullpath_filename = os.path.join(
            os.getcwd(),
            top_level_creation_folder,
            content_subfolder,
            f"{filename}{timestamp}.txt",
        )
    else:
        fullpath_filename = os.path.join(path, f"{filename}{timestamp}.txt")

    # Check if the directory exists, and create it if necessary
    os.makedirs(os.path.dirname(fullpath_filename), exist_ok=True)

    # Write the content to the file
    with open(fullpath_filename, "w") as file:
        file.write(content)


def save_best_chapter_to_output_file(
    content: str, full_path_and_file_name: str
) -> None:
    """
    Saves the best chapter content to a user-specified output file.

    Parameters:
        content (str): The chapter content to be saved into the file.
        full_path_and_file_name (str): The full path to the file where the best chapter content should be saved.

    Raises:
        AssertionError: If the `content` is not a string.
        Exception: If there was an error creating the directory or writing to the file.
    """
    # Ensure the input content is a string
    assert isinstance(content, str), "content must be a string"

    # Get the directory name from the full file path
    directory = os.path.dirname(full_path_and_file_name)

    # Check if the directory exists, and create it if necessary
    os.makedirs(directory, exist_ok=True)

    # Write the content to the file
    with open(full_path_and_file_name, "w") as file:
        file.write(content)


def dfs_collect_conversations(
    top_level_creation_folder: str,
    chain_of_thoughts_conversations_subfolder: str,
    outputs_config: DictConfig,
    node: ChapterNode,
    chain_of_thought: List[ChapterNode],
    chain_of_thought_index: int,
    save_chain_of_thoughts_conversations: Callable[
        [int, List[ChapterNode], str, Any], None
    ],
) -> int:
    """
    Private helper method that collects prompt messages from a given node and all its children (DFS).

    Parameters:
        top_level_creation_folder (str): The top-level creation folder.
        chain_of_thoughts_conversations_subfolder (str): The subfolder for chain of thoughts conversations.
        outputs_config (DictConfig): The configuration for output_artifacts.
        node (ChapterNode): The current node in the DFS traversal.
        chain_of_thought (List[ChapterNode]): The current chain of thought of nodes from the root to the current node.
        chain_of_thought_index (int): The number of completed chains of thought found so far.
        save_chain_of_thoughts_conversations (Callable): The function to save chain of thoughts conversations.

    Returns:
        int: The updated number of completed chains of thought.
    """

    # Append the current node's prompt messages to the chain_of_thought
    chain_of_thought.append(node)

    # Traverse all the children
    for child in node.children:
        chain_of_thought_index = dfs_collect_conversations(
            top_level_creation_folder=top_level_creation_folder,
            chain_of_thoughts_conversations_subfolder=chain_of_thoughts_conversations_subfolder,
            outputs_config=outputs_config,
            node=child,
            chain_of_thought=chain_of_thought,
            chain_of_thought_index=chain_of_thought_index,
            save_chain_of_thoughts_conversations=save_chain_of_thoughts_conversations,
        )

    if not node.children:
        dir_path = os.path.join(
            top_level_creation_folder,
            chain_of_thoughts_conversations_subfolder,
        )
        file_path = os.path.join(
            dir_path,
            f"{str(chain_of_thought_index)}.{outputs_config.chain_of_thoughts_conversations_output_format}",
        )

        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Prepare a dictionary of extra arguments
        extra_args = {"dark_mode": outputs_config.html_files_dark_mode}

        # Call the function with extra arguments
        save_chain_of_thoughts_conversations(
            chain_of_thought_index, chain_of_thought, file_path, **extra_args
        )
        chain_of_thought_index += 1

    # Pop the current node off the chain_of_thought before returning to the parent
    chain_of_thought.pop()

    return chain_of_thought_index


def save_chains_of_thoughts(
    top_level_creation_folder: str,
    chain_of_thoughts_conversations_subfolder: str,
    outputs_config: DictConfig,
    root: ChapterNode,
) -> None:
    """
    Save all "chains of thoughts" in the MCTS tree to markdown files.
    A "chain of thought" is essentially a path from the root to a leaf node in the MCTS tree that represents
    a complete creation process to generate a specific chapter.
    Each node in this path corresponds to a stage in this creation process.

    :return: None
    """
    # Select the appropriate "save_chain_of_thoughts_conversations" function
    save_chain_of_thoughts_conversations = (
        save_chain_of_thoughts_conversations_to_html
        if outputs_config.chain_of_thoughts_conversations_output_format == "html"
        else save_chain_of_thoughts_conversations_to_markdown
    )

    logger.info(
        f"I'm generating and saving chains_of_thoughts (i.e. each branch of the tree) for better redability"
    )
    # Check if the tree is not empty
    if not root:
        raise ValueError("The MCTS tree is empty.")

    # Start the DFS traversal from the root
    chain_of_thought_index = 1
    dfs_collect_conversations(
        top_level_creation_folder=top_level_creation_folder,
        chain_of_thoughts_conversations_subfolder=chain_of_thoughts_conversations_subfolder,
        outputs_config=outputs_config,
        node=root,
        chain_of_thought=[],
        chain_of_thought_index=chain_of_thought_index,
        save_chain_of_thoughts_conversations=save_chain_of_thoughts_conversations,
    )


def save_all_chapters_data_to_csv(
    top_level_creation_folder: str, results_subfolder: str, chapters_rows: List[Any]
) -> None:
    """
    Saves all chapters generated in the current creation session to a CSV file.

    :param chapters_rows: A list of chapter data.
    """
    save_to_file = os.path.join(
        top_level_creation_folder, results_subfolder, CHAPTER_ROWS
    )
    logger.info(
        f"Saving all chapters generated in current creation session to {save_to_file}"
    )
    try:
        n_chapters_df = DataFrame(chapters_rows)
        n_chapters_df.to_csv(save_to_file)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"An error occurred while saving all chapters generated on current creation session.\nException: {e}\n{tb}"
        )


def initialize_chapter_info_dict(**kwargs) -> Dict[str, Any]:
    """
    Initialize the chapter information dictionary with default values or provided key-value pairs.

    :param kwargs: Optional key-value pairs to override the default values.
    :return: A dictionary containing the chapter information.
    """
    # Default chapter information dictionary
    chapter_info_dict = {
        "chapter_number": 1,
        "total_num_chapters": 1,
        "general_error_messages": None,
        "chapter_critic_success": True,
        "chapter_critic_prompt_messages": [
            SystemMessage(content=""),
            HumanMessage(content=""),
        ],
        "chapter_critic_response_dict": {},
        "chapter_generator_success": False,
        "chapter_generator_prompt_messages": [
            SystemMessage(content=""),
            HumanMessage(content=""),
        ],
        "chapter_generator_response_dict": {},
    }

    # Update the chapter information dictionary with the provided key-value pairs
    chapter_info_dict.update(kwargs)

    return chapter_info_dict


def load_root_from_json(
    path_to_json_file: str, scoring_config: DictConfig
) -> Optional[ChapterNode]:
    """
    Load the root node from a JSON file.

    :param path_to_json_file: A string containing the path to the JSON file.
    :return: A ChapterNode object which will be set as the root node of the agent, or None if an error occurred.
    """
    try:
        # Open the JSON file and load the data
        with open(path_to_json_file, "r") as json_file:
            chapter_generator_response_dict = json.load(json_file)

        # Check that the loaded data is a dictionary, as expected
        if not isinstance(chapter_generator_response_dict, dict):
            raise TypeError(
                f"Data loaded from {path_to_json_file} is not a dictionary."
            )

        chapter_info_dict = initialize_chapter_info_dict(
            chapter_generator_response_dict=chapter_generator_response_dict
        )

        root_node = ChapterNode(scoring_config)
        root_node.set_chapter_dict(chapter_info_dict)

        return root_node

    except FileNotFoundError:
        logger.error(f"File {path_to_json_file} not found.")
        return None
    except json.JSONDecontentError:
        logger.error(f"Error decoding JSON from {path_to_json_file}.")
        return None
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"An error occurred while loading the root chapter from JSON.\nException: {e}\n{tb}"
        )
        return None


def load_root_from_content_file(
    path_to_content_file: str, scoring_config: DictConfig
) -> Optional[ChapterNode]:
    """
    Load the root node from a Chapter text file.

    :param path_to_content_file: A string containing the path to the Chapter text file.
    :return: A ChapterNode object which will be set as the root node of the agent, or None if an error occurred.
    """
    try:
        # Open the Chapter text file and read the data
        with open(path_to_content_file, "r") as content_file:
            content = content_file.read()

        # Initialize the chapter_info_dict with the loaded content
        chapter_generator_response_dict = {
            "rationale": "",
            "plan": "",
            "synpsis": "",
            "content": content,
        }
        chapter_info_dict = initialize_chapter_info_dict(
            chapter_generator_success=True,
            chapter_generator_response_dict=chapter_generator_response_dict,
        )

        root_node = ChapterNode(scoring_config)
        root_node.set_chapter_dict(chapter_info_dict)

        return root_node

    except FileNotFoundError:
        logger.error(f"File {path_to_content_file} not found.")
        return None
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"An error occurred while loading the root chapter from the Chapter text file.\nException: {e}\n{tb}"
        )
        return None
