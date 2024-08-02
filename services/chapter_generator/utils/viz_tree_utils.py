import glob
import os
import traceback
import numpy as np
import dill
import graphviz
import imageio
from PIL import Image
import re

from services.chapter_generator.utils.chapter_node import ChapterNode

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


DEFAULT_ROOT_NODE_FOLDER = "./stories/*_chapter_creation_data_*"
MCTS_TREE_FILE = "mcts_state.dill"
TREE_EVOLUTION_FOLDER = "tree_evolution"

# Global variable to keep track of the node with the highest chapter score
best_chapter_node = None
best_chapter_score = 0


def extract_number(filename) -> str:
    """
    Extracts number from filename.

    :param filename: filename string.
    :return: number found in filename as a string, or '00000' if no number is found.
    """
    match = re.search(r"\d+", filename)
    if match:
        # Pad the number with zeroes to ensure correct sorting
        return format(
            int(match.group()), "05d"
        )  # Pad to 5 digits - adjust as necessary
    else:
        return "00000"  # Default to '00000' if no number is found


def build_graph_viz_helper(node: ChapterNode, graph) -> None:
    """
    Helper function for building Graphviz graph.

    :param node: ChapterNode object.
    :param graph: Graphviz graph object.
    """
    global best_chapter_node
    global best_chapter_score

    if not node.expanded():
        return

    chapter_info_dict = node.chapter_info_dict or {}

    if node.parent is None:  # This is the root node
        label = "{Root - ID: %d}" % node.node_id
        color = "black"
        shape = "ellipse"
    else:
        chapter_score = node.get_chapter_quality_score() if chapter_info_dict else 0.0
        if chapter_score > best_chapter_score:
            best_chapter_score = chapter_score
            best_chapter_node = node

        # Build label for the node
        label = "{Node ID: %d | chapter_score: %.4f | visits: %d | parent ID: %s}" % (
            node.node_id,
            chapter_score,
            node.visit_count,
            node.parent.node_id,
        )

        # Set color for the node based on chapter score
        if chapter_score > 0.7:
            color = "green"
        elif chapter_score > 0.3:
            color = "yellow"
        else:
            color = "red"

        shape = "record"

    # Create the node
    graph.node(str(node.node_id), label=label, color=color, shape=shape)

    # Create edges to all children
    for child in node.children:
        graph.edge(str(node.node_id), str(child.node_id))
        build_graph_viz_helper(child, graph)

    # Once we finish building the graph, highlight the best chapter node
    if node.parent is None and best_chapter_node is not None:
        label = "{Node ID: %d | chapter_score: %.4f | Visits: %.d | parent ID: %s}" % (
            best_chapter_node.node_id,
            best_chapter_score,
            best_chapter_node.visit_count,
            best_chapter_node.parent.node_id,
        )
        color = "blue"
        shape = "record"
        graph.node(
            str(best_chapter_node.node_id),
            label=label,
            color=color,
            shape=shape,
            penwidth="3.0",
        )


def build_graph_viz(path_to_root_node_folder: str = None) -> bool:
    """
    Build a Graphviz graph from a root node and save it as a PDF file.

    :param path_to_root_node_folder: Path to the root node folder. If not provided, the most recent one is used.
    :retunr bool. True is creation of graph was successful
    """
    try:
        if path_to_root_node_folder is None:
            chapter_creation_folders = glob.glob(DEFAULT_ROOT_NODE_FOLDER)
            chapter_creation_folders.sort(key=os.path.getmtime, reverse=True)
            if chapter_creation_folders:
                path_to_root_node_folder = chapter_creation_folders[0]
            else:
                logger.info("No chapter creation folders found.")
                return False

        root_node_file = os.path.join(path_to_root_node_folder, MCTS_TREE_FILE)

        if os.path.exists(root_node_file):
            with open(root_node_file, "rb") as f:
                saved_data = dill.load(f)
                root = saved_data["root_node"]
                root.scoring_config = saved_data["scoring_config"]
            dot = graphviz.Digraph(node_attr={"shape": "record"})
            build_graph_viz_helper(root, dot)
            output_file = os.path.join(path_to_root_node_folder, "tree_viz")
            dot.format = "png"
            dot.render(filename=output_file, view=False, cleanup=True)
        else:
            logger.info(f"Previous chapter creation file not found: {root_node_file}")

    except Exception as e:
        tb = traceback.format_exc()
        logger.info(
            f"Could NOT load & visualize {root_node_file}.\nException: {e}\n{tb}"
        )
        return False


def animate_mcts_evolution(path_to_root_node_folder: str = None) -> bool:
    """
    Create an animated visualization of MCTS evolution.

    :param path_to_root_node_folder: Path to the root node folder. If not provided, the most recent one is used.
    :retunr bool. True is creation of video was successful
    """
    try:
        # If no folder was provided, find the most recently created one
        if path_to_root_node_folder is None:
            chapter_creation_folders = glob.glob(DEFAULT_ROOT_NODE_FOLDER)
            chapter_creation_folders.sort(key=os.path.getmtime, reverse=True)
            if chapter_creation_folders:
                path_to_root_node_folder = chapter_creation_folders[0]
            else:
                logger.info("No chapter creation folders found.")
                return False

        path_to_root_node_folder = os.path.join(
            path_to_root_node_folder, TREE_EVOLUTION_FOLDER
        )

        # Find all MCTS state files and sort them based on the extracted number
        mcts_files = sorted(
            glob.glob(os.path.join(path_to_root_node_folder, "*_mcts_state.dill")),
            key=extract_number,
        )

        # Create output directory for images
        output_directory = os.path.join(path_to_root_node_folder, "images")
        os.makedirs(output_directory, exist_ok=True)

        # Keep track of the maximum dimensions
        max_width = 0
        max_height = 0

        image_paths = []

        for i, file in enumerate(mcts_files):
            try:
                with open(file, "rb") as f:
                    data = dill.load(f)
                    root_node = data["root_node"]
                    root_node.scoring_config = data["scoring_config"]

                g = graphviz.Digraph(node_attr={"shape": "record"})
                build_graph_viz_helper(root_node, g)

                # Save the graph to a PNG file and add the image path to the list
                image_path = os.path.join(output_directory, f"graph_{i}")
                g.format = "png"
                g.render(filename=image_path, view=False, cleanup=True)
                image_path = f"{image_path}.png"
                # Update the maximum dimensions
                image = Image.open(image_path)
                width, height = image.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                image_paths.append(image_path)

            except Exception as e:
                tb = traceback.format_exc()
                logger.info(f"Could NOT load tree file {file}.\nException: {e}\n{tb}")

        # Pad the images to the maximum dimensions
        padded_images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            padded_image = Image.new("RGB", (max_width, max_height), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            padded_images.append(padded_image)

        # Create a MP4 video from the images
        writer = imageio.get_writer(
            os.path.join(output_directory, "tree_evolution.mp4"), fps=0.3
        )
        for padded_image in padded_images:
            writer.append_data(np.array(padded_image))
        writer.close()

    except Exception as e:
        tb = traceback.format_exc()
        logger.info(
            f"Could NOT load & visualize the tree evolution.\nException: {e}\n{tb}"
        )
        return False


if __name__ == "__main__":
    build_graph_viz()
