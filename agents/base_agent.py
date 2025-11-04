from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re

class BaseAgent:
    """Base class for all Gemini-powered agents with Chain-of-Thought support."""

    def __init__(self, api_key: str, model_name: str = "models/gemini-2.5-flash", temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    def think(self, prompt: str):
        """Run LLM reasoning and return response."""
        response = self.llm.invoke(prompt)
        return response.content.strip()

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

Format your response as:
<reasoning>
[Your detailed step-by-step thinking process here]
</reasoning>

<answer>
[Your final answer here]
</answer>
"""
        response = self.llm.invoke(cot_prompt)
        content = response.content.strip()
        
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else content
        
        if return_reasoning:
            return {
                "reasoning": reasoning,
                "answer": answer
            }
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
        responses = []
        
        for i in range(num_samples):
            cot_prompt = f"""
{prompt}

Think through this step-by-step. This is reasoning path {i+1}/{num_samples}.
Show your work and explain your reasoning clearly.

<reasoning>
[Your step-by-step thinking]
</reasoning>

<answer>
[Your final answer]
</answer>
"""
            response = self.llm.invoke(cot_prompt)
            content = response.content.strip()
            
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            
            if reasoning_match and answer_match:
                responses.append({
                    "reasoning": reasoning_match.group(1).strip(),
                    "answer": answer_match.group(1).strip()
                })
        
        return responses[0] if responses else {"reasoning": "", "answer": ""}