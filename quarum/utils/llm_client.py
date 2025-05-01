"""
LLM client module for interacting with language models.

This module provides a unified interface for making requests to
different language model providers with error handling, retry
logic, and logging.
"""

import time
import logging
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from quarum.utils.json_parsing import JSONExtractor

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with language models.

    This class provides a unified interface for making requests to
    different LLM providers, with support for retries, error handling,
    and response parsing.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4.1-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.0,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            temperature: Sampling temperature (0.0 = deterministic)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.llm = None
        self._setup_llm()

    def _setup_llm(self) -> None:
        """Set up the language model client."""
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("Error initializing LLM: %s", str(e))
            raise

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ) -> str:
        """
        Call the language model with a prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature_override: Optional temperature override

        Returns:
            Model response as string

        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Create prompt template
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))

        chat_prompt = ChatPromptTemplate.from_messages(messages)

        # Override temperature if specified
        if (
            temperature_override is not None
            and temperature_override != self.temperature
        ):
            original_llm = self.llm
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                temperature=temperature_override,
            )

        # Create chain
        chain = chat_prompt | self.llm

        # Try with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Call LLM
                response = chain.invoke({})

                # Get content from response
                content = (
                    response.content if hasattr(response, 'content') else str(response)
                )

                # Reset LLM if temperature was overridden
                if (
                    temperature_override is not None
                    and temperature_override != self.temperature
                ):
                    self.llm = original_llm

                return content

            except Exception as e:
                last_error = e
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, str(e))

                # Wait before retrying
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Progressive backoff

        # Reset LLM if temperature was overridden
        if (
            temperature_override is not None
            and temperature_override != self.temperature
        ):
            self.llm = original_llm

        # If we get here, all retries failed
        error_message = "All %d LLM call attempts failed. Last error: %s"
        logger.error(error_message, self.max_retries, str(last_error))
        raise RuntimeError(error_message % (self.max_retries, str(last_error)))

    def batch_call(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        parallel: bool = False,
    ) -> list[str]:
        """
        Call the language model with multiple prompts.

        Args:
            prompts: list of user prompts
            system_prompt: Optional system instruction
            parallel: Whether to make requests in parallel

        Returns:
            list of model responses
        """
        results = []

        if parallel:
            # Implement parallel calling (could use asyncio or threading)
            raise NotImplementedError("Parallel batch calling not yet implemented")
        else:
            # Serial processing
            for prompt in prompts:
                try:
                    result = self.call(prompt, system_prompt)
                    results.append(result)
                except Exception as e:
                    logger.error("Error in batch call: %s", str(e))
                    results.append(None)

        return results

    def call_with_json_extraction(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """
        Call LLM and extract JSON from the response.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction

        Returns:
            Extracted JSON object or None if extraction failed
        """

        # Call LLM
        response = self.call(prompt, system_prompt)

        # Extract JSON
        extractor = JSONExtractor()
        return extractor.extract(response)
