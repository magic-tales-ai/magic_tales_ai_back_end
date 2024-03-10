import copy
import traceback
from typing import Any, Dict
from omegaconf import DictConfig

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ChapterNode:
    """
    A class that represents a node in the Monte Carlo Tree Search (MCTS) tree.
    Each node holds a chapter and its associated information.
    """

    # Class variable to keep track of the number of instances created
    node_counter = 0

    def __init__(
        self,
        scoring_config: DictConfig,
        chapter_info_dict: Dict[str, Any] = None,
        parent=None,
        prior=None,
    ):
        """
        Initialize a ChapterNode.

        :param chapter_info_dict: A dictionary containing chapter information.
        :param prior: The prior probability of the node.
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = []
        self.scoring_config = scoring_config
        self.chapter_info_dict = (
            copy.deepcopy(chapter_info_dict) if chapter_info_dict is not None else {}
        )
        self.parent = parent

        # Increment the node counter and assign it to this instance
        # self.node_id = uuid.uuid4()
        ChapterNode.node_counter += 1
        self.node_id = ChapterNode.node_counter

    def set_chapter_dict(self, chapter_info_dict: Dict[str, Any]) -> None:
        """
        Sets the chapter_info_dict attribute of the node.

        :param chapter_info_dict: A dictionary containing chapter information.
        :return: None
        """
        self.chapter_info_dict = copy.deepcopy(chapter_info_dict)

    def is_empty(self) -> bool:
        """
        Checks if the node contains any actual content (and its key) in the chapter info dict

        :return: bool
        """
        return (
            self.chapter_info_dict.get("chapter_generator_response_dict", {}).get(
                "content", ""
            )
            == ""
        )

    def value(self) -> float:
        """
        Classic MCTS average value for the node

        :return: The average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        """
        Check if the node is expanded.

        :return: True if the node is expanded, False otherwise.
        """
        return len(self.children) > 0

    def get_float_value(self, key: str, default: float = 0.0) -> float:
        """Utility function to safely get float values from the chapter_info_dict."""
        value = self.chapter_info_dict.get("chapter_critic_response_dict", {}).get(
            key, default
        )
        try:
            return float(value)
        except ValueError:
            return default

    def calculate_score_for_metric(self, weight: float, key: str) -> float:
        """Calculate the weighted score for a specific metric."""
        return weight * self.get_float_value(key)

    def get_chapter_quality_score(self) -> float:
        """
        Calculate the weighted score of the node.

        :return: float - the weighted score of the node.
        :raises: Exception if an error occurs during calculation.
        """
        if self._is_invalid_node():
            return 0.0

        try:
            metrics = self._get_quality_metrics()
            (
                weighted_scores,
                total_weight,
            ) = self._calculate_weighted_scores_and_total_weight(
                metrics, self.scoring_config
            )

            if total_weight > 1.0:
                weighted_scores, total_weight = self._normalize_weighted_scores(
                    weighted_scores, total_weight
                )

            return self._calculate_final_quality_score(weighted_scores, total_weight)

        except Exception as e:
            self._handle_quality_score_error(e)
            raise e

    def _is_invalid_node(self) -> bool:
        return (
            self.parent is None
            or self.chapter_info_dict is {}
            or not self.chapter_info_dict.get("chapter_generator_success", False)
            or not self.chapter_info_dict.get("chapter_critic_success", False)
        )

    def _get_quality_metrics(self) -> list:
        return [
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

    def _calculate_weighted_scores_and_total_weight(
        self, metrics: list, scoring_weights: object
    ) -> tuple:
        weighted_scores = [
            self.calculate_score_for_metric(
                getattr(scoring_weights, f"weight_{metric}"), metric
            )
            for metric in metrics
        ]
        total_weight = sum(
            getattr(scoring_weights, f"weight_{metric}") for metric in metrics
        )
        return weighted_scores, total_weight

    def _normalize_weighted_scores(
        self, weighted_scores: list, total_weight: float
    ) -> tuple:
        normalized_weights = [score / total_weight for score in weighted_scores]
        return normalized_weights, 1.0

    def _calculate_final_quality_score(
        self, weighted_scores: list, total_weight: float
    ) -> float:
        total_weight = total_weight if total_weight != 0.0 else 1.0
        return sum(weighted_scores) / total_weight

    def _handle_quality_score_error(self, exception: Exception) -> None:
        tb = traceback.format_exc()
        logger.error(
            f"Error calculating chapter quality score.\nException: {exception}\n{tb}"
        )

    def expand(self, child) -> None:
        """
        Expand the node by adding a new child.

        :param child: A node to become the child.
        """
        child.parent = self
        self.children.append(child)

    def __repr__(self) -> str:
        """
        Represent the ChapterNode as a string.

        :return: A string representation of the ChapterNode.
        """
        parent_id = "" if self.parent is None else self.parent.node_id

        return "Node ID: %d, chapter_score: %s, visits: %d, parent ID: %s" % (
            self.node_id,
            self.get_chapter_quality_score(),
            self.visit_count,
            parent_id,
        )
