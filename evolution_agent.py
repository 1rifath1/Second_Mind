import logging
import requests
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvolutionAgent:
    """
    Evolution Agent refines hypotheses and ideas based on:
    1. Web data from the Reflection Agent
    2. Similar past interactions from the Proximity Agent
    3. Ranked keywords from the Ranking Agent

    It uses the Groq API to generate refined hypotheses that are more accurate
    and relevant based on all available information.
    """

    def __init__(self, api_key=None):
        """
        Initialize the Evolution Agent with a Groq API key.

        Args:
            api_key: Groq API key for text generation
        """
        self.api_key = api_key or "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"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info("Evolution Agent initialized.")

    def refine_hypothesis(self,
                          original_query: str,
                          original_hypothesis: str,
                          ranked_keywords: List[Tuple[str, Dict[str, Any]]],
                          search_results: List[Dict[str, Any]],
                          proximity_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Refine the hypothesis based on search results, keywords, and proximity data.

        Args:
            original_query: The original user query
            original_hypothesis: The hypothesis from the Generation Agent
            ranked_keywords: Keywords ranked by the Ranking Agent
            search_results: Enriched search results
            proximity_data: Information from similar past interactions

        Returns:
            Dictionary with refined hypothesis and explanation
        """
        # Prepare the prompt for refinement
        prompt = self._build_refinement_prompt(
            original_query,
            original_hypothesis,
            ranked_keywords,
            search_results,
            proximity_data
        )

        # Get refined hypothesis using Groq API
        refined_results = self._generate_refined_hypothesis(prompt)

        # Extract hypothesis and explanation
        refined_hypothesis = refined_results.get("refined_hypothesis", "")
        explanation = refined_results.get("explanation", "")
        improvements = refined_results.get("improvements", [])

        return {
            "original_hypothesis": original_hypothesis,
            "refined_hypothesis": refined_hypothesis,
            "explanation": explanation,
            "improvements": improvements,
            "version": 1,  # For tracking refinement cycles
            "timestamp": time.time()
        }

    def _build_refinement_prompt(self,
                                 original_query: str,
                                 original_hypothesis: str,
                                 ranked_keywords: List[Tuple[str, Dict[str, Any]]],
                                 search_results: List[Dict[str, Any]],
                                 proximity_data: Dict[str, Any] = None) -> str:
        """
        Build a prompt for the Groq API to generate a refined hypothesis.

        Args:
            original_query: The original user query
            original_hypothesis: The hypothesis from the Generation Agent
            ranked_keywords: Keywords ranked by the Ranking Agent
            search_results: Enriched search results
            proximity_data: Information from similar past interactions

        Returns:
            Prompt string for the Groq API
        """
        # Start with system instruction
        prompt = """You are an expert scientific hypothesis refinement system. Your goal is to improve a scientific hypothesis by incorporating:
1. Insights from web search results
2. Keywords ranked by relevance
3. Information from similar past queries (if available)

For your output, provide:
1. A refined, improved hypothesis (2-3 sentences, clearly labeled as "REFINED HYPOTHESIS:")
2. A brief explanation of what was improved (labeled as "EXPLANATION:")
3. A list of 2-3 specific improvements made (labeled as "IMPROVEMENTS:")

Make the hypothesis more precise, evidence-based, and scientifically sound.
"""

        # Add original query and hypothesis
        prompt += f"\n\nORIGINAL QUERY: {original_query}"
        prompt += f"\n\nORIGINAL HYPOTHESIS: {original_hypothesis}"

        # Add ranked keywords
        if ranked_keywords:
            prompt += "\n\nRANKED KEYWORDS (by relevance):"
            for i, (keyword, data) in enumerate(ranked_keywords[:5], 1):
                frequency = data.get("total_frequency", 0)
                prompt += f"\n{i}. {keyword} (frequency: {frequency})"

        # Add top search results
        if search_results:
            prompt += "\n\nTOP SEARCH RESULTS:"
            for i, result in enumerate(search_results[:5], 1):
                title = result.get("title", "")
                extracted_keywords = result.get("extracted_keywords", "")
                prompt += f"\n{i}. {title}"
                if extracted_keywords:
                    prompt += f"\n   Keywords: {extracted_keywords}"

        # Add proximity data if available
        if proximity_data:
            prompt += "\n\nINSIGHTS FROM SIMILAR PAST QUERIES:"

            # Add similar queries
            similar_queries = proximity_data.get("similar_queries", [])
            if similar_queries:
                prompt += "\nSimilar past queries:"
                for i, query_data in enumerate(similar_queries[:2], 1):
                    similarity = query_data.get("similarity", 0) * 100
                    prompt += f"\n- \"{query_data['query']}\" ({similarity:.1f}% similar)"

            # Add common keywords
            common_keywords = proximity_data.get("common_keywords", {})
            if common_keywords:
                prompt += "\nCommon keywords from similar interactions:"
                top_keywords = list(common_keywords.items())[:3]
                for keyword, frequency in top_keywords:
                    prompt += f"\n- {keyword} (appeared in {frequency} similar interactions)"

            # Add adaptation suggestions
            suggestions = proximity_data.get("adaptation_suggestions", [])
            if suggestions:
                prompt += "\nSuggestions based on similar past interactions:"
                for suggestion in suggestions[:2]:
                    prompt += f"\n- {suggestion}"

        return prompt

    def _generate_refined_hypothesis(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a refined hypothesis using the Groq API.

        Args:
            prompt: The prompt for refinement

        Returns:
            Dictionary with refined hypothesis, explanation, and improvements
        """
        try:
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 800  # Increased from 500 to ensure complete responses
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]

            # Log the full response for debugging
            logger.debug(f"Groq API response: {response_text}")

            # Extract refined hypothesis, explanation, and improvements
            refined_hypothesis = self._extract_section(
                response_text, "REFINED HYPOTHESIS")
            explanation = self._extract_section(response_text, "EXPLANATION")

            # Extract improvements as a list
            improvements_text = self._extract_section(
                response_text, "IMPROVEMENTS")
            improvements = self._parse_improvements(improvements_text)

            # If any section is empty, log a warning
            if not refined_hypothesis:
                logger.warning(
                    "Failed to extract refined hypothesis from response")
                # Use a different pattern or approach
                if "refined hypothesis" in response_text.lower():
                    # Try a more lenient pattern
                    pattern = r"(?:refined hypothesis:?|the refined hypothesis is:?)(.*?)(?:\n\n|$)"
                    match = re.search(pattern, response_text,
                                      re.DOTALL | re.IGNORECASE)
                    refined_hypothesis = match.group(
                        1).strip() if match else ""

            if not explanation:
                logger.warning("Failed to extract explanation from response")

            if not improvements:
                logger.warning("Failed to extract improvements from response")
                # Try to extract numbered points if available
                improvements = re.findall(
                    r'\d+\.\s+(.*?)(?=\d+\.|$)', response_text, re.DOTALL)
                improvements = [imp.strip()
                                for imp in improvements if imp.strip()]

            # As a fallback, if sections are still empty but we have a response
            if response_text and (not refined_hypothesis or not explanation or not improvements):
                # Split by double newlines and try to extract sections
                sections = response_text.split("\n\n")

                if not refined_hypothesis and len(sections) > 0:
                    for section in sections:
                        if "refined" in section.lower() and len(section) > 30:
                            refined_hypothesis = section
                            break

                if not explanation and len(sections) > 1:
                    for section in sections:
                        if "explanation" in section.lower() or "improved" in section.lower():
                            explanation = section
                            break

                if not improvements and len(sections) > 2:
                    for section in sections:
                        if "improvement" in section.lower() or "change" in section.lower():
                            # Try to split by lines and extract potential improvements
                            potential_improvements = [line.strip() for line in section.split("\n")
                                                      if line.strip() and not line.lower().startswith("improvement")]
                            if potential_improvements:
                                improvements = potential_improvements
                            break

            return {
                "refined_hypothesis": refined_hypothesis or "Unable to generate a refined hypothesis.",
                "explanation": explanation or "No explanation provided.",
                "improvements": improvements or ["No specific improvements identified."]
            }

        except Exception as e:
            logger.error(f"Error generating refined hypothesis: {e}")
            return {
                "refined_hypothesis": "Error generating refined hypothesis.",
                "explanation": f"An error occurred: {str(e)}",
                "improvements": []
            }

    def _extract_section(self, text: str, section_name: str) -> str:
        """
        Extract a section from the response text.

        Args:
            text: The response text
            section_name: The name of the section to extract

        Returns:
            The extracted section text
        """
        pattern = rf"{section_name}:?(.*?)(?:\n\n|$|\n[A-Z ]+:)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _parse_improvements(self, improvements_text: str) -> List[str]:
        """
        Parse improvements text into a list of improvements.

        Args:
            improvements_text: The improvements section text

        Returns:
            List of improvements
        """
        if not improvements_text:
            return []

        # Try to find numbered or bulleted items
        items = re.findall(
            r'(?:^|\n)(?:\d+\.|\-|\*)\s*(.*?)(?=(?:\n(?:\d+\.|\-|\*))|$)', improvements_text, re.DOTALL)

        if items:
            return [item.strip() for item in items]

        # Fallback: split by newlines
        return [line.strip() for line in improvements_text.split('\n') if line.strip()]

    def further_refine(self,
                       original_query: str,
                       current_refinement: Dict[str, Any],
                       new_search_results: List[Dict[str, Any]] = None,
                       feedback: str = None) -> Dict[str, Any]:
        """
        Further refine an already refined hypothesis based on new information or feedback.

        Args:
            original_query: The original user query
            current_refinement: The current refinement data
            new_search_results: New search results (optional)
            feedback: Feedback on the current refinement (optional)

        Returns:
            Updated refinement data
        """
        # Get the current hypothesis and version
        current_hypothesis = current_refinement.get("refined_hypothesis", "")
        version = current_refinement.get("version", 1)

        # Build prompt for further refinement
        prompt = """You are an expert scientific hypothesis refinement system. Your goal is to further improve an already refined hypothesis by incorporating:
1. New search results (if provided)
2. Feedback on the current hypothesis (if provided)

For your output, provide:
1. A further improved hypothesis (2-3 sentences, clearly labeled as "REFINED HYPOTHESIS:")
2. A brief explanation of what was improved (labeled as "EXPLANATION:")
3. A list of 2-3 specific improvements made (labeled as "IMPROVEMENTS:")

Make the hypothesis more precise, evidence-based, and scientifically sound.
"""

        # Add original query and current hypothesis
        prompt += f"\n\nORIGINAL QUERY: {original_query}"
        prompt += f"\n\nCURRENT HYPOTHESIS (Version {version}): {current_hypothesis}"

        # Add new search results if available
        if new_search_results:
            prompt += "\n\nNEW SEARCH RESULTS:"
            for i, result in enumerate(new_search_results[:3], 1):
                title = result.get("title", "")
                extracted_keywords = result.get("extracted_keywords", "")
                prompt += f"\n{i}. {title}"
                if extracted_keywords:
                    prompt += f"\n   Keywords: {extracted_keywords}"

        # Add feedback if available
        if feedback:
            prompt += f"\n\nFEEDBACK ON CURRENT HYPOTHESIS: {feedback}"

        # Generate refined hypothesis
        refined_results = self._generate_refined_hypothesis(prompt)

        # Update refinement data
        return {
            "original_hypothesis": current_refinement.get("original_hypothesis", ""),
            "refined_hypothesis": refined_results.get("refined_hypothesis", current_hypothesis),
            "explanation": refined_results.get("explanation", ""),
            "improvements": refined_results.get("improvements", []),
            "version": version + 1,  # Increment version
            "timestamp": time.time()
        }

    def refine_with_focus(self,
                          original_query: str,
                          original_hypothesis: str,
                          focus_area: str,
                          ranked_keywords: List[Tuple[str,
                                                      Dict[str, Any]]] = None,
                          search_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refine a hypothesis with a specific focus area.

        Args:
            original_query: The original user query
            original_hypothesis: The hypothesis to refine
            focus_area: The specific aspect to focus on (e.g., "methodology", "applications")
            ranked_keywords: Keywords ranked by the Ranking Agent (optional)
            search_results: Enriched search results (optional)

        Returns:
            Refined hypothesis data
        """
        # Build prompt with focus area
        prompt = f"""You are an expert scientific hypothesis refinement system. Your goal is to improve a scientific hypothesis specifically focusing on the "{focus_area}" aspect.

For your output, provide:
1. A refined, improved hypothesis (2-3 sentences, clearly labeled as "REFINED HYPOTHESIS:")
2. A brief explanation of what was improved in relation to the {focus_area} (labeled as "EXPLANATION:")
3. A list of 2-3 specific improvements made (labeled as "IMPROVEMENTS:")

Make the hypothesis more precise, evidence-based, and scientifically sound, with special attention to {focus_area}.
"""

        # Add original query and hypothesis
        prompt += f"\n\nORIGINAL QUERY: {original_query}"
        prompt += f"\n\nORIGINAL HYPOTHESIS: {original_hypothesis}"

        # Add ranked keywords if available
        if ranked_keywords:
            prompt += "\n\nRANKED KEYWORDS (by relevance):"
            for i, (keyword, data) in enumerate(ranked_keywords[:5], 1):
                frequency = data.get("total_frequency", 0)
                prompt += f"\n{i}. {keyword} (frequency: {frequency})"

        # Add top search results if available
        if search_results:
            prompt += "\n\nRELEVANT SEARCH RESULTS:"
            for i, result in enumerate(search_results[:3], 1):
                title = result.get("title", "")
                extracted_keywords = result.get("extracted_keywords", "")
                prompt += f"\n{i}. {title}"
                if extracted_keywords:
                    prompt += f"\n   Keywords: {extracted_keywords}"

        # Generate refined hypothesis
        refined_results = self._generate_refined_hypothesis(prompt)

        return {
            "original_hypothesis": original_hypothesis,
            "refined_hypothesis": refined_results.get("refined_hypothesis", ""),
            "explanation": refined_results.get("explanation", ""),
            "improvements": refined_results.get("improvements", []),
            "focus_area": focus_area,
            "version": 1,
            "timestamp": time.time()
        }

    def compare_hypotheses(self,
                           original_hypothesis: str,
                           refined_hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare the original hypothesis with multiple refinements.

        Args:
            original_hypothesis: The original hypothesis
            refined_hypotheses: List of refined hypothesis data

        Returns:
            Comparison analysis
        """
        # Build prompt for comparison
        prompt = """You are an expert scientific hypothesis evaluator. Your task is to compare an original hypothesis with multiple refinements of it.

For your output, provide:
1. A brief analysis of each hypothesis version (original and refinements)
2. A comparison of how each refinement improved upon the original
3. A recommendation for which version is scientifically strongest, with reasoning
4. A suggestion for any further improvements that could be made

Structure your analysis clearly with labeled sections.
"""

        # Add original hypothesis
        prompt += f"\n\nORIGINAL HYPOTHESIS: {original_hypothesis}"

        # Add refined hypotheses
        for i, refinement in enumerate(refined_hypotheses, 1):
            version = refinement.get("version", i)
            focus = refinement.get("focus_area", "general")
            hypothesis = refinement.get("refined_hypothesis", "")
            prompt += f"\n\nREFINEMENT {version} (focus: {focus}): {hypothesis}"

        # Generate comparison analysis
        try:
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1000
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            comparison_text = result["choices"][0]["message"]["content"]

            # Extract sections from comparison
            analysis = self._extract_multi_section(
                comparison_text, "ANALYSIS", "COMPARISON")
            comparison = self._extract_multi_section(
                comparison_text, "COMPARISON", "RECOMMENDATION")
            recommendation = self._extract_multi_section(
                comparison_text, "RECOMMENDATION", "FURTHER IMPROVEMENTS")
            further_improvements = self._extract_multi_section(
                comparison_text, "FURTHER IMPROVEMENTS", None)

            return {
                "original_hypothesis": original_hypothesis,
                "refinement_count": len(refined_hypotheses),
                "analysis": analysis,
                "comparison": comparison,
                "recommendation": recommendation,
                "further_improvements": further_improvements,
                "full_comparison": comparison_text
            }

        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return {
                "original_hypothesis": original_hypothesis,
                "refinement_count": len(refined_hypotheses),
                "error": f"Error generating comparison: {str(e)}"
            }

    def _extract_multi_section(self, text: str, start_section: str, end_section: str = None) -> str:
        """
        Extract a section from the response text with multiple sections.

        Args:
            text: The response text
            start_section: The name of the section to start extraction
            end_section: The name of the section to end extraction (optional)

        Returns:
            The extracted section text
        """
        if end_section:
            pattern = rf"{start_section}:?(.*?)(?:{end_section}:)"
        else:
            pattern = rf"{start_section}:?(.*?)$"

        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def generate_alternative_hypotheses(self,
                                        query: str,
                                        original_hypothesis: str,
                                        search_results: List[Dict[str,
                                                                  Any]] = None,
                                        count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple alternative hypotheses for the same query.

        Args:
            query: The original query
            original_hypothesis: The original hypothesis
            search_results: Enriched search results (optional)
            count: Number of alternatives to generate

        Returns:
            List of alternative hypotheses with explanations
        """
        # Build prompt for alternative generation
        prompt = f"""You are an expert scientific hypothesis generator. Generate {count} distinct alternative hypotheses for the following query, each taking a different perspective or approach to the topic.

QUERY: {query}

ORIGINAL HYPOTHESIS: {original_hypothesis}

For each alternative hypothesis, provide:
1. The hypothesis itself (2-3 sentences)
2. A brief explanation of how it differs from the original
3. What specific perspective or approach it represents

Generate hypotheses that are scientifically sound but explore different angles than the original.
"""

        # Add search results if available
        if search_results:
            prompt += "\n\nRELEVANT SEARCH RESULTS:"
            for i, result in enumerate(search_results[:5], 1):
                title = result.get("title", "")
                prompt += f"\n{i}. {title}"

        # Generate alternatives
        try:
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,  # Higher temperature for more diversity
                "max_tokens": 1500
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            alternatives_text = result["choices"][0]["message"]["content"]

            # Parse alternatives
            alternatives = []
            alternative_sections = re.split(
                r'Alternative (?:Hypothesis )?#?\d+:', alternatives_text)

            # The first element might be empty or introductory text
            for section in alternative_sections[1:]:
                if not section.strip():
                    continue

                hypothesis = ""
                explanation = ""
                perspective = ""

                # Extract hypothesis
                hypothesis_match = re.search(
                    r'(.*?)(?:Explanation:|Differs from original:|Perspective:)', section, re.DOTALL | re.IGNORECASE)
                if hypothesis_match:
                    hypothesis = hypothesis_match.group(1).strip()

                # Extract explanation
                explanation_match = re.search(
                    r'(?:Explanation:|Differs from original:)(.*?)(?:Perspective:|$)', section, re.DOTALL | re.IGNORECASE)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()

                # Extract perspective
                perspective_match = re.search(
                    r'(?:Perspective:|Approach:)(.*?)(?:$|\n\n)', section, re.DOTALL | re.IGNORECASE)
                if perspective_match:
                    perspective = perspective_match.group(1).strip()

                alternatives.append({
                    "hypothesis": hypothesis,
                    "explanation": explanation,
                    "perspective": perspective
                })

            # Ensure we return the requested number (or fewer if parsing failed)
            return alternatives[:count]

        except Exception as e:
            logger.error(f"Error generating alternative hypotheses: {e}")
            return [{
                "hypothesis": f"Error generating alternative hypothesis: {str(e)}",
                "explanation": "",
                "perspective": ""
            }]

    def merge_hypotheses(self,
                         query: str,
                         hypotheses: List[str],
                         strengths: List[str] = None) -> Dict[str, Any]:
        """
        Merge multiple hypotheses into a single, stronger hypothesis.

        Args:
            query: The original query
            hypotheses: List of hypotheses to merge
            strengths: List of strengths for each hypothesis (optional)

        Returns:
            Merged hypothesis data
        """
        if len(hypotheses) < 2:
            return {
                "merged_hypothesis": hypotheses[0] if hypotheses else "",
                "explanation": "No merge performed as fewer than 2 hypotheses were provided.",
                "strengths_incorporated": []
            }

        # Build prompt for merging
        prompt = """You are an expert scientific hypothesis synthesizer. Your task is to merge multiple hypotheses into a single, stronger hypothesis that incorporates the best elements of each.

For your output, provide:
1. A merged hypothesis (2-3 sentences, clearly labeled as "MERGED HYPOTHESIS:")
2. An explanation of how the merge was performed (labeled as "EXPLANATION:")
3. A list of strengths incorporated from each input hypothesis (labeled as "STRENGTHS INCORPORATED:")

Create a cohesive, scientifically sound hypothesis that represents the best synthesis of the inputs.
"""

        # Add original query
        prompt += f"\n\nQUERY: {query}"

        # Add hypotheses
        for i, hypothesis in enumerate(hypotheses, 1):
            prompt += f"\n\nHYPOTHESIS {i}: {hypothesis}"
            if strengths and i <= len(strengths) and strengths[i-1]:
                prompt += f"\nStrength: {strengths[i-1]}"

        # Generate merged hypothesis
        try:
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 800
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            merge_text = result["choices"][0]["message"]["content"]

            # Extract sections
            merged_hypothesis = self._extract_section(
                merge_text, "MERGED HYPOTHESIS")
            explanation = self._extract_section(merge_text, "EXPLANATION")
            strengths_incorporated_text = self._extract_section(
                merge_text, "STRENGTHS INCORPORATED")

            # Parse strengths
            strengths_incorporated = self._parse_improvements(
                strengths_incorporated_text)

            return {
                "merged_hypothesis": merged_hypothesis,
                "explanation": explanation,
                "strengths_incorporated": strengths_incorporated
            }

        except Exception as e:
            logger.error(f"Error merging hypotheses: {e}")
            return {
                "merged_hypothesis": "Error merging hypotheses.",
                "explanation": f"An error occurred: {str(e)}",
                "strengths_incorporated": []
            }
