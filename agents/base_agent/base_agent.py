from langchain_google_genai import ChatGoogleGenerativeAI
import re
import warnings


class BaseAgent:
    """Base class for all Gemini-powered agents with Chain-of-Thought support."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-2.5-flash",
        temperature: float = 0.3,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    def _extract_content(self, response):
        """
        Extract text content from LLM response, handling both string and list formats.

        Args:
            response: LLM response object

        Returns:
            String content
        """
        content = response.content

        # Handle list of content blocks (multimodal responses)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Handle dict format: {"type": "text", "text": "..."}
                    if block.get("type") == "text" and "text" in block:
                        text_parts.append(block["text"])
                    # Handle other dict formats
                    elif "text" in block:
                        text_parts.append(block["text"])
                elif isinstance(block, str):
                    # Handle string blocks
                    text_parts.append(block)
                elif hasattr(block, "text"):
                    # Handle objects with text attribute
                    text_parts.append(block.text)
            content = "\n".join(text_parts)

        # Ensure we have a string
        if not isinstance(content, str):
            content = str(content)

        return content.strip()

    def think(self, prompt: str):
        """Deprecated: simple LLM invoke wrapper.

        Kept for backwards compatibility. Prefer `think_with_cot` or the
        Chain-of-Thought APIs. This wrapper emits a deprecation warning
        and forwards to the underlying implementation to avoid changing
        runtime behaviour.
        """
        warnings.warn(
            "BaseAgent.think() is deprecated; use think_with_cot() instead",
            DeprecationWarning,
        )
        return self._deprecated_think_impl(prompt)

    def _deprecated_think_impl(self, prompt: str):
        """Original implementation of `think` kept as a private deprecated impl."""
        response = self.llm.invoke(prompt)
        return self._extract_content(response)

    def think_with_cot(self, prompt: str, return_reasoning: bool = False):
        """
        Run LLM reasoning with Chain-of-Thought approach.

        Args:
            prompt: The input prompt
            return_reasoning: If True, returns both reasoning and answer

        Returns:
            If return_reasoning=False: Just the final answer
            If return_reasoning=True: Dict with 'reasoning' and 'answer'
        """
        cot_prompt = f"""
{prompt}

Think through this step-by-step:
1. First, analyze the problem and identify key constraints
2. Consider different approaches and their trade-offs
3. Reason through the implications of each decision
4. Arrive at a well-justified conclusion

IMPORTANT: Format your response EXACTLY as shown below, with opening and closing tags on their own lines:

<reasoning>
[Your detailed step-by-step thinking process here]
</reasoning>

<answer>
[Your final answer here]
</answer>

Do not include any text before <reasoning> or after </answer>.
"""
        response = self.llm.invoke(cot_prompt)
        content = self._extract_content(response)

        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else content

        if return_reasoning:
            return {"reasoning": reasoning, "answer": answer}
        return answer

    def think_with_self_consistency(self, prompt: str, num_samples: int = 3):
        """
        Use self-consistency: generate multiple reasoning paths and select most consistent answer.

        Args:
            prompt: The input prompt
            num_samples: Number of reasoning paths to generate

        Returns:
            The most consistent answer with reasoning
        """
        warnings.warn(
            "BaseAgent.think_with_self_consistency() is deprecated; use repeated calls to think_with_cot() and aggregate externally",
            DeprecationWarning,
        )
        return self._deprecated_think_with_self_consistency_impl(prompt, num_samples)

    def _deprecated_think_with_self_consistency_impl(
        self, prompt: str, num_samples: int = 3
    ):
        """Deprecated implementation kept for compatibility.

        Returns the first valid reasoning/answer pair generated (if any).
        """
        responses = []
        for i in range(num_samples):
            cot_prompt = f"""
{prompt}

Think through this step-by-step. This is reasoning path {i+1}/{num_samples}.
Show your work and explain your reasoning clearly.

IMPORTANT: Format your response EXACTLY as shown below:

<reasoning>
[Your step-by-step thinking]
</reasoning>

<answer>
[Your final answer]
</answer>
"""
            response = self.llm.invoke(cot_prompt)
            content = self._extract_content(response)

            reasoning_match = re.search(
                r"<reasoning>(.*?)</reasoning>", content, re.DOTALL
            )
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)

            if reasoning_match and answer_match:
                responses.append(
                    {
                        "reasoning": reasoning_match.group(1).strip(),
                        "answer": answer_match.group(1).strip(),
                    }
                )

        return responses[0] if responses else {"reasoning": "", "answer": ""}
