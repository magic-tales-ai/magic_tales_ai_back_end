import math
from typing import Dict

from services.chapter_generator.utils.chapter_node import ChapterNode
from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        """
        Initialize the MinMaxStats object.
        """
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        """
        Update the min-max values with a new value.

        :param value: The new value.
        """
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        """
        Normalize a value.

        :param value: The value to normalize.
        :return: The normalized value.
        """
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score(
    parent: ChapterNode,
    child: ChapterNode,
    min_max_stats: MinMaxStats,
    mcts_params: Dict,
) -> float:
    """
    Calculate the UCB (Upper Confidence Bound) score for a child node.

    :param parent: The parent node.
    :param child: The child node.
    :param min_max_stats: The MinMaxStats object for normalization.
    :return: The UCB score of the child node.
    """
    try:
        pb_c = (
            math.log(
                (parent.visit_count + mcts_params.pb_c_base + 1) / mcts_params.pb_c_base
            )
            + mcts_params.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        # Uniform prior for continuous action space
        if mcts_params.node_prior == "uniform":
            if len(parent.children) > 0:
                prior_score = pb_c * (1 / len(parent.children))
            else:
                prior_score = 0.0
        elif mcts_params.node_prior == "density":
            total_prior = sum([child.prior for child in parent.children.values()])
            if total_prior > 0:
                prior_score = pb_c * (child.prior / total_prior)
            else:
                prior_score = 0.0
        else:
            raise ValueError("{} is unknown prior option, choose uniform or density")

        if child.visit_count > 0:
            # The value score is the average value of the node
            value_score = min_max_stats.normalize(child.get_chapter_quality_score())
        else:
            value_score = 0.0

    except Exception as e:
        logger.info(f"Error calculating UCB score for child {child}: {e}")
        prior_score = 0.0
        value_score = 0.0

    return prior_score + value_score


#  ****** Stat Update Methods ******


def simple_average_update(node: ChapterNode, performance_metric: float) -> float:
    """
    Update the value of a node using the simple average of its current value and a new performance metric.

    :param node: The node to update.
    :param performance_metric: The new performance metric.
    :return: The updated value of the node.
    """
    return (node.value() * node.visit_count + performance_metric) / (
        node.visit_count + 1
    )


def max_performance_update(node: ChapterNode, performance_metric: float) -> float:
    """
    Update the value of a node to be the maximum of its current value and a new performance metric.

    :param node: The node to update.
    :param performance_metric: The new performance metric.
    :return: The updated value of the node.
    """
    return max(node.get_chapter_quality_score(), performance_metric)


def weighted_average_update(
    node: ChapterNode, performance_metric: float, weight: float
) -> float:
    """
    Update the value of a node using a weighted average of its current value and a new performance metric.

    :param node: The node to update.
    :param performance_metric: The new performance metric.
    :param weight: The weight for the new performance metric.
    :return: The updated value of the node.
    """
    return (node.value() * node.visit_count + weight * performance_metric) / (
        node.visit_count + weight
    )
