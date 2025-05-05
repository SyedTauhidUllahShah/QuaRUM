"""
Base pipeline module for domain modeling.

This module defines the abstract Pipeline class that serves
as the foundation for domain modeling pipelines.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Results from a pipeline execution.

    This dataclass captures the outputs, metrics, and messages
    from a pipeline execution for later analysis and reporting.
    """

    # Whether the pipeline completed successfully
    success: bool = False

    # Time taken to execute the pipeline
    execution_time: float = 0.0

    # Outputs produced by the pipeline
    outputs: dict[str, Any] = field(default_factory=dict)

    # Metrics collected during execution
    metrics: dict[str, Any] = field(default_factory=dict)

    # Messages and logs
    messages: list[str] = field(default_factory=list)

    # Phase results
    phase_results: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: str) -> None:
        """Add a message to the results."""
        self.messages.append(message)

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to the results."""
        self.metrics[name] = value

    def add_output(self, name: str, output: Any) -> None:
        """Add an output to the results."""
        self.outputs[name] = output

    def add_phase_result(self, phase_name: str, result: Any) -> None:
        """Add a phase result to the results."""
        self.phase_results[phase_name] = result


class Pipeline(ABC):
    """
    Abstract base class for domain modeling pipelines.

    This class defines the common interface and functionality for
    domain modeling pipelines, including setup, execution, and
    result handling.
    """

    def __init__(self, name: str):
        """
        Initialize the pipeline.

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.messages = []
        self.metrics = {}
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """
        Set up the pipeline with the given parameters.

        Args:
            **kwargs: Pipeline-specific parameters

        Returns:
            True if setup successful, False otherwise
        """

    @abstractmethod
    def execute(self, **kwargs) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            **kwargs: Pipeline-specific parameters

        Returns:
            Result of the pipeline execution
        """

    def _start_execution(self) -> None:
        """Record the start time of execution."""
        self.start_time = time.time()
        logger.info("Starting %s pipeline", self.name)

    def _end_execution(self) -> float:
        """
        Record the end time of execution.

        Returns:
            Execution time in seconds
        """
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        logger.info("Completed %s pipeline in %.2f seconds", self.name, execution_time)
        return execution_time

    def add_message(self, message: str) -> None:
        """
        Add a message to the pipeline log.

        Args:
            message: Message to add
        """
        self.messages.append(message)
        logger.info(message)

    def add_warning(self, message: str) -> None:
        """
        Add a warning message to the pipeline log.

        Args:
            message: Warning message to add
        """
        warning = f"WARNING: {message}"
        self.messages.append(warning)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        """
        Add an error message to the pipeline log.

        Args:
            message: Error message to add
        """
        error = f"ERROR: {message}"
        self.messages.append(error)
        logger.error(message)

    def add_metric(self, name: str, value: Any) -> None:
        """
        Add a metric to the pipeline.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        logger.debug("Metric %s: %s", name, value)

    def create_result(
        self, success: bool, outputs: dict[str, Any] = None
    ) -> PipelineResult:
        """
        Create a pipeline result.

        Args:
            success: Whether the pipeline executed successfully
            outputs: dictionary of outputs

        Returns:
            PipelineResult object
        """
        # Calculate execution time
        execution_time = 0.0
        if self.start_time is not None:
            if self.end_time is None:
                self._end_execution()
            execution_time = self.end_time - self.start_time

        # Create result object
        result = PipelineResult(
            success=success,
            execution_time=execution_time,
            outputs=outputs or {},
            metrics=self.metrics.copy(),
            messages=self.messages.copy(),
        )

        return result
