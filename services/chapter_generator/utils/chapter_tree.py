from typing import List
import numpy
import inspect
from omegaconf import DictConfig

from services.chapter_generator.utils.chapter_node import ChapterNode
from services.chapter_generator.utils.mcts_utils import (
    MinMaxStats,
    max_performance_update,
    simple_average_update,
    weighted_average_update,
    ucb_score,
)
from .saving_loading_utils import initialize_chapter_info_dict

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


# Defining the MCTS ChapterTree class
class ChapterTree:
    def __init__(self, mcts_config: DictConfig, node_scoring_config: DictConfig):
        """
        Initialize the ChapterTree with a root node.

        :param root: The root node of the tree.
        """
        self.mcts_config = mcts_config
        self.node_scoring_config = node_scoring_config
        self.reset()

        # Initialize available update methods
        self.update_methods_available = {
            "simple_average": simple_average_update,
            "max_performance": max_performance_update,
            "weighted_average": weighted_average_update,
            # Add any other update methods here...
        }

        # Validate update method
        if self.mcts_config.update_method not in self.update_methods_available:
            raise ValueError(f"Invalid update method: {self.update_method}")

        # Assign update method
        self.update_method = self.update_methods_available[
            self.mcts_config.update_method
        ]
        self.update_method_params = self.mcts_config.update_method_params

    def success(self) -> bool:
        return (
            self.best_chapter is not None
            and self.best_chapter.get_chapter_quality_score()
            >= self.mcts_config.quality_score_threshold
        )

    def reset(self) -> None:
        self.root = None
        self.best_chapter: ChapterNode = None
        self.min_max_stats = MinMaxStats()
        self.max_tree_depth = 0

    def select_leaf_node(self, search_path: List[ChapterNode]) -> ChapterNode:
        """
        Select a leaf node from the tree while balancing exploitation and exploration.

        :param search_path: The path from the root to the selected node.
        :return: The selected leaf node.
        """
        node = self.root
        while node.expanded():
            node = self._select_child(node, self.min_max_stats)
            search_path.append(node)
        return node

    def expand_tree(self, node: ChapterNode) -> ChapterNode:
        """
        Expand the tree with a new node. If the node is the root, use the initial content (example) if available, to expand it.

        :param node: The ChapterNode that should be expanded.
        """
        if node.parent is None:  # This is the root node
            return self.expand_root()
        else:
            new_child_node = ChapterNode(scoring_config=self.node_scoring_config)
            node.expand(new_child_node)
            return new_child_node

    def backpropagate(self, search_path: List[ChapterNode]) -> None:
        """
        At the end of a Chapter creation and evaluation/critique, we propagate the score all the way up the tree
        to the root.

        :param search_path: The path of nodes from the root to the leaf.
        :param min_max_stats: The MinMaxStats object.
        """
        performance_metric = search_path[-1].get_chapter_quality_score()
        logger.info(
            f"I'm backpropagating the 'Chapter Quality Score (CQS)' of the node ({performance_metric}) using the update method selected"
        )
        for node in reversed(search_path):
            node.value_sum += performance_metric
            node.visit_count += 1
            quality_score = node.get_chapter_quality_score()
            self.min_max_stats.update(quality_score)

            # Update the performance_metric to continue the backpropagation
            kwargs = {}
            if "performance_metric" in inspect.signature(self.update_method).parameters:
                kwargs["performance_metric"] = performance_metric
            kwargs.update(self.update_method_params)
            performance_metric = self.update_method(node, **kwargs)

            # Update best chapter if this node's performance is higher
            if (
                self.best_chapter is None
                or quality_score > self.best_chapter.get_chapter_quality_score()
            ):
                self.best_chapter = node

        self.max_tree_depth = max(self.max_tree_depth, len(search_path))

    def expand_root(self) -> ChapterNode:
        """
        Expand the root node with a new child that contains the initial content (example) if available.
        If the root node does not have any initial content, expand it without any content.
        """
        # Initialize the chapter_info_dict for the new child
        chapter_info_dict = initialize_chapter_info_dict()

        # Check if the root node has any initial content
        if "content" in self.root.chapter_info_dict.get(
            "chapter_generator_response_dict", {}
        ):
            # Extract the initial content from the root node
            initial_code = self.root.chapter_info_dict[
                "chapter_generator_response_dict"
            ]["content"]
            # Set the initial content for the new child
            chapter_info_dict["chapter_generator_response_dict"] = {
                "content": initial_code
            }
        else:
            # No initial content found in the root node, log a warning message
            logger.warning(
                "Root node does not contain any initial chapter. Expanding root without any content."
            )

        # Create a new child node and add it as a child of the root
        child_node = ChapterNode(scoring_config=self.node_scoring_config)
        child_node.set_chapter_dict(chapter_info_dict)
        self.root.expand(child_node)
        return child_node

    def select_child(self, node) -> ChapterNode:
        """
        Select the child with the highest UCB score.

        :param node: The parent node from which to select a child.
        :return: The selected child node.
        """
        if len(node.children) < (node.visit_count + 1) ** self.mcts_config.pw_alpha:
            logger.info(f"Progressive widening: Adding a child to node {node.node_id}.")
            return self.expand_tree(node)

        max_ucb = max(
            ucb_score(node, child, self.min_max_stats, self.mcts_config)
            for child in node.children
        )
        child = numpy.random.choice(
            [
                child
                for child in node.children
                if numpy.isclose(
                    ucb_score(node, child, self.min_max_stats, self.mcts_config),
                    max_ucb,
                )
            ]
        )
        return child

    def _check_for_low_performance(self) -> bool:
        """
        Check if the performance of the best chapter is below a certain threshold and if the depth of the search tree has exceeded
        a certain limit.

        Returns:
            bool: True if the performance is low and the tree depth is high, False otherwise.
        """
        quality_score = self.best_chapter.get_chapter_quality_score()
        if (
            quality_score < self.mcts_config.regrow_threshold_quality_score
            and self.max_tree_depth > self.mcts_config.depth_limit_for_performance
        ):
            num_root_children = len(self.root.children)
            num_total_children = ChapterNode.node_counter
            logger.info(
                f"I've detected low chapter performance. Current quality score is {quality_score} and our threshold is {self.mcts_config.regrow_threshold_quality_score}. The tree is now {self.max_tree_depth} levels deep. The root node has {num_root_children} children and the tree has {num_total_children} nodes."
            )
            return True

        return False

    def _generate_fresh_chapters(self) -> None:
        """
        Generate new chapters from scratch (without previous content or critique) if low chapter performance is detected. These new
        chapters are added as children of the root node of the search tree.
        """
        logger.info(
            f"I'm generating new chapters from scratch (no previous content nor critic) since I've detected low chapter performance"
        )
        # Generate new chapters from scratch and add them as children of the root
        for _ in range(self.mcts_config.additional_root_children):
            self.expand_root()

    def check_performance_and_generate_fresh_chapters(self) -> None:
        """
        Check for low performance and generate fresh chapters if necessary.

        If the low performance check indicates that the current chapter is underperforming,
        this method will trigger the generation of fresh chapters to replace it.
        """
        # Low performance check
        if self._check_for_low_performance():
            self._generate_fresh_chapters()

    def get_best_chapter(self, node=None) -> None:
        """
        Traverses the MCTS tree and returns the node (i.e., the chapter) with the highest average value.

        We use a Depth-First Search (DFS) to traverse the tree. DFS visits the child nodes before the sibling nodes.
        This is efficient for our use case because it doesn't require extra space proportional to the depth of the tree.

        :param node: The node from where to start the search. Usually, it's the root node of the tree.
        :return: None (Internal assigment of the best chapter).
        """
        node = self.root if node is None else node

        # Initialize the best chapter and maximum value
        best_chapter = node
        max_value = node.get_chapter_quality_score()

        # Stack for DFS
        stack = [node]

        while stack:
            node = stack.pop()
            quality_score = node.get_chapter_quality_score()
            # Update the best chapter and maximum value if the current node's value is greater
            if quality_score > max_value:
                max_value = quality_score
                best_chapter = node

            # Push the children of the current node into the stack
            stack.extend(node.children)

        logger.info(
            f"Best chapter from previous chapter creation round: {best_chapter}"
        )
        self.best_chapter = best_chapter
