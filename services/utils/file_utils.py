import os
import json
import time
import traceback
import logging
import collections
from typing import List, Dict
import re
import html
from datetime import datetime


from services.chapter_generator.utils.chapter_node import ChapterNode


from typing import List, Dict, Optional, Union
from services.utils.log_utils import get_logger


# Get a logger instance for this module
logger = get_logger(__name__)


def datetime_converter(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(o).__name__)


async def convert_user_info_to_json_files(
    data: Dict, save_path: str, max_num_files: int = 20
) -> List[str]:
    """
    Converts user data, profiles, and stories grouped by profile into JSON files.
    Limits the number of files created.

    Args:
        data (Dict): The knowledge base data including user info, profiles, and stories.
        save_path (str): The directory path to save the JSON files.
        max_num_files (int): The maximum number of files to be created.

    Returns:
        List[str]: The list of file paths for the created JSON files.
    """
    files_paths = []
    files_created = 0

    try:
        os.makedirs(save_path, exist_ok=True)

        # User info
        if data.get("user_info") and files_created < max_num_files:
            user_info_path = os.path.join(save_path, f"user_info_{data['user_info'].id}.json")
            with open(user_info_path, "w", encoding="utf-8") as file:
                json.dump(data["user_info"].to_dict(), file, ensure_ascii=False, indent=4, default=datetime_converter)
            files_paths.append(user_info_path)
            files_created += 1

        # Profiles and stories
        for profile in data.get("profiles", []):
            if files_created >= max_num_files:
                break

            profile_data = profile.to_dict()
            stories_for_profile = [
                story.to_dict() for story in data.get("stories", []) if story.profile_id == profile.id
            ]

            if stories_for_profile:
                profile_data["stories"] = stories_for_profile

            profile_path = os.path.join(save_path, f"profile_{profile.id}_info_and_stories.json")
            with open(profile_path, "w", encoding="utf-8") as file:
                json.dump(profile_data, file, ensure_ascii=False, indent=4, default=datetime_converter)
            files_paths.append(profile_path)
            files_created += 1

    except Exception as e:
        logging.exception(f"Error in converting data to files: {e}")
        raise

    return files_paths

async def convert_user_info_to_md_files(data: Dict, save_path: str) -> List[str]:
    """
    Converts the knowledge base data into markdown-formatted text files.

    Args:
        data (Dict): The knowledge base data.

    Returns:
        List[str]: Paths to the created knowledge base text files.
    """
    file_paths = []
    try:
        # Ensure the save folder exists
        os.makedirs(save_path, exist_ok=True)

        # Convert user data to a text file with markdown formatting
        user_file_path = os.path.join(
            save_path, f"user_{data['user_info']['user_id']}.md"
        )
        with open(user_file_path, "w", encoding="utf-8") as file:
            file.write(format_markdown(data["user_info"], "User Info"))
        file_paths.append(user_file_path)

        # Convert profiles to text files with markdown formatting
        for profile in data["profiles"]:
            profile_file_path = os.path.join(
                save_path, f"profile_{profile['profile_id']}.md"
            )
            with open(profile_file_path, "w", encoding="utf-8") as file:
                file.write(format_markdown(profile, "Profile"))
            file_paths.append(profile_file_path)

        # Convert stories to text files with markdown formatting
        for story in data["stories"]:
            story_file_path = os.path.join(save_path, f"story_{story['story_id']}.md")
            with open(story_file_path, "w", encoding="utf-8") as file:
                file.write(format_markdown(story, "Story"))
            file_paths.append(story_file_path)

    except Exception as e:
        raise RuntimeError(f"Error in converting data to files: {e}")

    return file_paths


def format_markdown(self, data: Dict, title: str) -> str:
    """
    Formats a dictionary of data into a markdown string.

    Args:
        data (Dict): The data to format.
        title (str): The title for the markdown section.

    Returns:
        str: A markdown-formatted string.
    """
    markdown_text = f"## {title}\n"
    for key, value in data.items():
        markdown_text += f"- **{key.capitalize()}**: {value}\n"
    return markdown_text


def get_latest_story_directory(stories_root_dir: str) -> Union[str, None]:
    """
    Get the path to the most recently modified story directory.

    Args:
        stories_root_dir (str): The top-level directory containing story directories.

    Returns:
        Union[str, None]: The path to the latest story directory or None if no story directories are found.
    """
    # make sure we have a "stories_root_dir" folder
    os.makedirs(stories_root_dir, exist_ok=True)

    story_dirs = [
        d
        for d in os.listdir(stories_root_dir)
        if os.path.isdir(os.path.join(stories_root_dir, d))
    ]
    if not story_dirs:
        return None

    # Sort directories by their creation time
    story_dirs.sort(
        key=lambda x: os.path.getmtime(os.path.join(stories_root_dir, x)), reverse=True
    )

    return os.path.join(stories_root_dir, story_dirs[0])


def create_new_story_directory(
    stories_root_dir: str, subfolders: Dict[str, str]
) -> Optional[str]:
    """
    Creates the output directory for storing a single story and its related assets.

    Args:
        stories_root_dir (str): The root directory where all stories will be saved.
        subfolders (Dict[str, str]): A dictionary defining the structure of subfolders.

    Returns:
        Optional[str]: The path to the newly created story directory or None if an error occurs.
    """

    try:
        # Generate a timestamp for folder naming
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Create the top-level folder for this particular story
        top_level_story_folder = os.path.join(stories_root_dir, timestamp)
        os.makedirs(top_level_story_folder, exist_ok=True)
        logger.info(f"Created top-level story folder: {top_level_story_folder}")
        images_subfolder = subfolders.get("images", "images")

        # Create necessary subfolders
        for subfolder in subfolders.values():
            subfolder_path = os.path.join(top_level_story_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            logger.info(f"Created subfolder: {subfolder_path}")

        return top_level_story_folder, images_subfolder

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            f"Failed to create top-level story folder.\nException: {e}\nTraceback: {tb}"
        )
        return None


def save_agent_memory_to_disk(agent_memory, directory, filename):
    """Save agent memory to disk.

    Args:
        agent_memory: Agent's memory object.
        directory (str): The directory where the memory will be saved.
        filename (str): The filename where the memory will be saved.
    """
    # Extract the messages from the agent's memory
    messages = [message.content for message in agent_memory.messages]

    # Create the full path for the file
    filepath = os.path.join(directory, filename)

    # Save the messages to the file
    with open(filepath, "w") as file:
        json.dump(messages, file, indent=4)

    logger.info(f"Agent memory saved to: {filepath}")


def save_prompts_to_disk(prompts, directory, filename):
    """Save image prompts to disk.

    Args:
        prompts (List[str]): List of image prompts.
        directory (str): The directory where the prompts will be saved.
        filename (str): The filename where the prompts will be saved.
    """
    # Create the full path for the file
    filepath = os.path.join(directory, filename)

    # Save the prompts to the file
    with open(filepath, "w") as file:
        json.dump(prompts, file, indent=4)

    logger.info(f"Image prompts saved to: {filepath}")


def is_sequence(obj):
    """
    Returns:
      True if the sequence is a collections.Sequence and not a string.
    """
    return isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str)


def pack_varargs(args):
    """
    Pack *args or a single list arg as list

    def f(*args):
        arg_list = pack_varargs(args)
        # arg_list is now packed as a list
    """
    assert isinstance(args, tuple), "please input the tuple `args` as in *args"
    if len(args) == 1 and is_sequence(args[0]):
        return args[0]
    else:
        return args


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))


def f_join(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath


def load_text(*fpaths, by_lines=False):
    with open(f_join(*fpaths), "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


def get_color_map(dark_mode: bool) -> dict:
    """
    Get color map based on the dark_mode flag.

    :param dark_mode: A boolean flag indicating if the dark mode is enabled or not.
    :return: A dictionary with the color mapping.
    """
    if dark_mode:
        return {
            "generator_system": "87CEEB",
            "generator_human": "32CD32",
            "generator_response": "FA8072",
            "critic_system": "00BFFF",
            "critic_human": "00FF00",
            "critic_response": "FFA07A",
        }
    else:
        return {
            "generator_system": "0000CC",
            "generator_human": "006400",
            "generator_response": "8B0000",
            "critic_system": "4169E1",
            "critic_human": "228B22",
            "critic_response": "B22222",
        }


def format_critic_response_html(input_dict):
    if not isinstance(input_dict, dict):
        input_dict = str(input_dict)

    formatted_parts = []

    for key, value in input_dict.items():
        if "score" in key:
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong> {value}</p>"
            )
        elif key == "rationale":
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong> {value}</p>"
            )
        elif key == "critique":
            lines = re.split(r"\s(?=\d+\))", value)
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong>"
                + "".join(f"<li>{line}</li>" for line in lines)
            )

    return "<br><br>".join(formatted_parts)


def format_generator_response_html(input_dict):
    if not isinstance(input_dict, dict):
        input_dict = str(input_dict)

    formatted_parts = []

    for key, value in input_dict.items():
        if key == "rationale":
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong> {value}</p>"
            )
        elif key == "plan" or key == "used_chapter_primitives":
            lines = re.split(r"\s(?=\d+\))", value)
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong>"
                + "".join(f"<li>{line}</li>" for line in lines)
            )
        elif key == "content":
            code_block = "<pre><content>" + html.escape(value) + "</content></pre>"
            formatted_parts.append(
                f"<p><strong>{key.capitalize()}:</strong><br>{code_block}</p>"
            )

    return "<br><br>".join(formatted_parts)


def save_chain_of_thoughts_conversations_to_html(
    chain_of_thought_index: int,
    chain_of_thought: List[ChapterNode],
    file_path: str,
    **kwargs,
) -> None:
    dark_mode = kwargs.get("dark_mode", False)

    color_map = get_color_map(dark_mode)

    html = "<h1>A Chain of Thought Number " + str(chain_of_thought_index) + "</h1>\n"
    creation_cycle = 1

    for node in chain_of_thought:
        if node.chapter_info_dict is None:
            continue
        chapter_info_dict = node.chapter_info_dict

        # Generator Messages
        generator_messages = chapter_info_dict.get(
            "chapter_generator_prompt_messages", [None, None]
        )
        generator_response_dict = chapter_info_dict.get(
            "chapter_generator_response_dict", {}
        )

        html += "<h2>Creation Cycle Number " + str(creation_cycle) + ":</h2>\n"
        creation_cycle += 1

        html += "<h3>Generator:</h3>\n"

        html += "<h4>System Message:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["generator_system"]
            + "'>"
            + generator_messages[0].content.replace("\n", "<br>")
            + "</p>\n"
        )

        html += "<h4>Human Message:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["generator_human"]
            + "'>"
            + generator_messages[1].content.replace("\n", "<br>")
            + "</p>\n"
        )

        html += "<h4>Response:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["generator_response"]
            + "'>"
            + format_generator_response_html(generator_response_dict)
            + "</p>\n"
        )

        # Critic Messages
        critic_messages = chapter_info_dict.get(
            "chapter_critic_prompt_messages", [None, None]
        )
        critic_response_dict = chapter_info_dict.get("chapter_critic_response_dict", {})

        html += "<h3>Critic:</h3>\n"

        html += "<h4>System Message:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["critic_system"]
            + "'>"
            + critic_messages[0].content.replace("\n", "<br>")
            + "</p>\n"
        )

        html += "<h4>Human Message:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["critic_human"]
            + "'>"
            + critic_messages[1].content.replace("\n", "<br>")
            + "</p>\n"
        )

        html += "<h4>Response:</h4>\n"
        html += (
            "<p style='color:#"
            + color_map["critic_response"]
            + "'>"
            + format_critic_response_html(critic_response_dict)
            + "</p>\n"
        )

        # Separator between creation cycles
        html += "<hr>\n"

    html = "<html>\n<body>\n" + html + "</body>\n</html>"

    try:
        with open(file_path, "w") as file:
            file.write(html)
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(
            "An error occurred when trying to save the HTML file.\nException: "
            + str(e)
            + "\n"
            + tb
        )


def format_critic_response_markdown(input_dict):
    if not isinstance(input_dict, dict):
        input_dict = str(input_dict)

    formatted_parts = []

    for key, value in input_dict.items():
        if "score" in key:
            formatted_parts.append(f"**{key.capitalize()}**: {value}")
        elif key == "rationale":
            lines = value.split("\n")
            formatted_parts.append(
                f"**{key.capitalize()}**:\n" + "\n".join(f"{line}" for line in lines)
            )
        elif key == "critique":
            lines = re.split(r"(\d+\))", value)
            formatted_parts.append(
                f"**{key.capitalize()}**:" + "".join(f"\n{line}" for line in lines)
            )

    return "\n\n".join(formatted_parts)


def format_generator_response_markdown(input_dict):
    if not isinstance(input_dict, dict):
        input_dict = str(input_dict)

    formatted_parts = []

    for key, value in input_dict.items():
        if key == "rationale" or key == "synopsis" or key == "content":
            lines = value.split("\n")
            formatted_parts.append(
                f"**{key.capitalize()}**:\n" + "\n".join(f"{line}" for line in lines)
            )
        elif key == "plan":
            lines = re.split(r"(\d+\))", value)
            formatted_parts.append(
                f"**{key.capitalize()}**:" + "".join(f"\n{line}" for line in lines)
            )

    return "\n\n".join(formatted_parts)


def save_chain_of_thoughts_conversations_to_markdown(
    chain_of_thought_index: int,
    chain_of_thought: List[ChapterNode],
    file_path: str,
    **kwargs,
) -> None:
    # Ignore kwargs and implement the function

    md = f"# A Chain of Thought Number {chain_of_thought_index}\n"
    creation_cycle = 1

    for node in chain_of_thought:
        if node.parent is None or node.chapter_info_dict == {}:
            continue
        chapter_info_dict = node.chapter_info_dict

        # Generator Messages
        generator_messages = chapter_info_dict.get(
            "chapter_generator_prompt_messages", ["", ""]
        )
        generator_response_dict = chapter_info_dict.get(
            "chapter_generator_response_dict", {}
        )

        md += f"\n## Creation Cycle Number {creation_cycle}\n"
        creation_cycle += 1

        md += "\n### Generator\n"

        md += "\n#### System Message:\n"
        md += (
            "-"
            if generator_messages[0] == ""
            else generator_messages[0].content.replace("\n", "  \n")
        )

        md += "\n#### Human Message:\n"
        md += (
            "-"
            if generator_messages[1] == ""
            else generator_messages[1].content.replace("\n", "  \n")
        )

        md += "\n#### Response:\n"
        md += format_generator_response_markdown(generator_response_dict)

        # Critic Messages
        critic_messages = chapter_info_dict.get(
            "chapter_critic_prompt_messages", ["", ""]
        )
        critic_response_dict = chapter_info_dict.get("chapter_critic_response_dict", {})

        md += "\n### Critic\n"

        md += "\n#### System Message:\n"
        md += (
            "-"
            if critic_messages[0] == ""
            else critic_messages[0].content.replace("\n", "  \n")
        )

        md += "\n#### Human Message:\n"
        md += (
            "-"
            if critic_messages[0] == ""
            else critic_messages[1].content.replace("\n", "  \n")
        )

        md += "\n#### Response:\n"
        md += format_critic_response_markdown(critic_response_dict)

        # Separator between creation cycles
        md += "\n---\n"

    try:
        with open(file_path, "w") as file:
            file.write(md)
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(
            "An error occurred when trying to save the Markdown file.\nException: "
            + str(e)
            + "\n"
            + tb
        )
