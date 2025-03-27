import logging
import json
import datetime
import time
import random
import requests
import re
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMetaReviewAgent:
    """
    Enhanced Meta-Review Agent evaluates the entire Second Mind system's process,
    automatically generating hypotheses, analyzing coherence, ranking keywords,
    and evaluating the entire process.
    """

    def __init__(self, api_key="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"):
        """
        Initialize the Enhanced Meta-Review Agent.
        """
        self.api_key = api_key
        self.process_metrics = {}
        self.historical_metrics = []
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Research topics provided by the user
        self.research_topics = [
            "Quantum Learning",
            "Machine Unlearning",
            "Sustainable software design",
            "Common sense reasoning in Agentic AI"
        ]

        logger.info("Enhanced Meta-Review Agent initialized.")

    def generate_hypothesis(self, query: str) -> Dict[str, Any]:
        """
        Generate a hypothesis using the Groq API.

        Args:
            query: The research topic

        Returns:
            Dictionary with the generated hypothesis
        """
        logger.info(f"Generating hypothesis for: {query}")

        formatted_prompt = f"Generate a concise scientific hypothesis about: {query}. Keep it under 2-3 sentences."

        try:
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": formatted_prompt}],
                "temperature": 0.7,
                "max_tokens": 200,
                "seed": random.randint(1, 10000)
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            hypothesis = result["choices"][0]["message"]["content"]

            return {
                "query": query,
                "hypothesis": hypothesis,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error generating hypothesis: {e}")
            return {
                "query": query,
                "hypothesis": f"Error generating hypothesis: {str(e)}",
                "timestamp": time.time()
            }

    def check_coherence(self, hypothesis: str, query: str) -> Dict[str, Any]:
        """
        Check the coherence of a hypothesis using a simplified coherence checker.

        Args:
            hypothesis: The hypothesis to check
            query: The original query

        Returns:
            Dictionary with coherence analysis results
        """
        logger.info(f"Checking coherence for hypothesis: {hypothesis[:50]}...")

        sentences = re.split(r'(?<=[.!?])\s+', hypothesis)

        # Calculate various factors used to determine coherence
        structure = {
            "has_hypothesis_structure": self._check_hypothesis_structure(hypothesis),
            "has_causal_language": self._check_causal_language(hypothesis),
            "has_scientific_terms": self._check_scientific_terms(hypothesis)
        }
        grammar_issues = self._check_grammar(hypothesis, sentences)
        relevance = {
            "relevant_to_prompt": self._check_relevance(hypothesis, query),
            "relevant_keywords": self._extract_relevant_keywords(hypothesis, query)
        }

        # Use an ideal sentence count factor (between 1 and 4 sentences)
        sentence_factor = (1 <= len(sentences) <= 4)
        grammar_factor = (len(grammar_issues) == 0)

        # Compute coherence score from several factors
        coherence_factors = [
            structure["has_hypothesis_structure"],
            structure["has_causal_language"],
            structure["has_scientific_terms"],
            relevance["relevant_to_prompt"],
            sentence_factor,
            grammar_factor
        ]
        coherence_score = round(
            sum(1 for factor in coherence_factors if factor) / len(coherence_factors) * 10, 1)

        if coherence_score >= 8:
            verdict = "Excellent"
        elif coherence_score >= 6:
            verdict = "Good"
        elif coherence_score >= 4:
            verdict = "Acceptable"
        else:
            verdict = "Needs improvement"

        return {
            "coherence_score": coherence_score,
            "verdict": verdict,
            "relevant_keywords": relevance["relevant_keywords"],
            "structure_analysis": structure,
            "grammar_issues": grammar_issues,
            "relevance_analysis": relevance
        }

    def _check_hypothesis_structure(self, text: str) -> bool:
        """Check if the text has a scientific hypothesis structure."""
        patterns = [
            r'\b(hypothesize|hypothesis)\b',
            r'\b(predict|prediction)\b',
            r'\b(suggest|propose)\b',
            r'\b(may|might|could)\s+\w+\b',
            r'\b(potential(ly)?|possible|possibly)\b',
            r'\bif\s+.+\s+then\b'
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)

    def _check_causal_language(self, text: str) -> bool:
        """Check if the text contains causal language."""
        patterns = [
            r'\b(cause|caused|causing)\b',
            r'\b(effect|affects|affected|affecting)\b',
            r'\b(impact|impacts|impacted|impacting)\b',
            r'\b(influence|influences|influenced|influencing)\b',
            r'\b(lead|leads|leading)\s+to\b',
            r'\b(result|results|resulted|resulting)\s+(in|from)\b',
            r'\bcorrelat(e|es|ed|ion)\b',
            r'\brelationship\b',
            r'\bassociat(e|ed|ion)\b'
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)

    def _check_scientific_terms(self, text: str) -> bool:
        """Check if the text contains scientific terminology."""
        patterns = [
            r'\b\w+tion\b',
            r'\b\w+sis\b',
            r'\b\w+cal\b',
            r'\b\w+ism\b',
            r'\b\w+ology\b',
            r'\b\w+meter\b',
            r'\bdata\b', r'\banalysis\b', r'\bresearch\b', r'\bstudy\b',
            r'\bexperiment\b', r'\btheory\b', r'\bmethod\b'
        ]
        return any(re.search(pattern, text.lower()) for pattern in patterns)

    def _check_grammar(self, text: str, sentences: List[str]) -> List[str]:
        """Perform basic grammar checks."""
        issues = []
        # Check for repeated words
        repeated = re.findall(r'\b(\w+)\s+\1\b', text.lower())
        if repeated:
            issues.append(f"Repeated words: {', '.join(repeated)}")
        # Check for overly long sentences (more than 40 words)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 40:
                issues.append(
                    f"Sentence {i+1} is very long ({len(sentence.split())} words)")
        # Check if each sentence starts with a capital letter
        for i, sentence in enumerate(sentences):
            if sentence and not sentence[0].isupper():
                issues.append(
                    f"Sentence {i+1} does not begin with a capital letter")
        return issues

    def _check_relevance(self, hypothesis: str, prompt: str) -> bool:
        """Check if the hypothesis is relevant to the prompt."""
        if prompt.lower() in hypothesis.lower():
            return True
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        hypothesis_words = set(re.findall(r'\b\w{4,}\b', hypothesis.lower()))
        return len(prompt_words.intersection(hypothesis_words)) >= 1

    def _extract_relevant_keywords(self, hypothesis: str, prompt: str) -> List[str]:
        """
        Use the Groq API to extract keywords that capture the connection between the hypothesis and the prompt.

        Args:
            hypothesis: The hypothesis
            prompt: The original prompt

        Returns:
            List of extracted keywords
        """
        logger.info("Extracting relevant keywords...")

        try:
            system_message = (
                "You are an expert text analyzer. Given a hypothesis and its prompt, "
                "extract a comma-separated list of 5 relevant keywords that highlight the connection between them. "
                "Only output the keywords, with no extra commentary."
            )
            user_message = f"Hypothesis: {hypothesis}\nPrompt: {prompt}\nKeywords:"

            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.3,
                "max_tokens": 60
            }

            response = requests.post(
                self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            keywords_output = result["choices"][0]["message"]["content"]

            # Parse the comma-separated keywords
            keywords = [kw.strip()
                        for kw in keywords_output.split(',') if kw.strip()]
            return keywords[:5]  # Limit to 5 keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def generate_search_results(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Generate simulated search results for testing.

        Args:
            keywords: List of keywords to use for search result generation

        Returns:
            List of simulated search results
        """
        logger.info(
            f"Generating simulated search results for keywords: {keywords}")

        # Sample article titles by research domain
        domains = {
            "medicine": [
                "Journal of Medical Research: Advances in Treatment Methods",
                "Clinical Trials Review: Statistical Analysis of Outcomes",
                "Medical Hypotheses: New Approaches to Disease Management",
                "Frontiers in Medicine: Emerging Diagnostic Techniques",
                "The New England Journal of Medicine: Randomized Control Studies"
            ],
            "technology": [
                "IEEE Transactions: Novel Algorithms for Data Processing",
                "Journal of Artificial Intelligence: Neural Network Applications",
                "Computational Research Quarterly: Optimizing System Architecture",
                "Technology Review: Breakthrough Innovations in Computing",
                "Journal of Data Science: Statistical Learning Methods"
            ],
            "psychology": [
                "Journal of Cognitive Psychology: Mental Processing Patterns",
                "Behavioral Science Review: Understanding Human Development",
                "Psychological Bulletin: Meta-Analysis of Therapy Outcomes",
                "Brain and Behavior: Neural Correlates of Cognition",
                "Journal of Social Psychology: Interpersonal Dynamics"
            ],
            "environment": [
                "Environmental Science & Technology: Climate Adaptation Strategies",
                "Nature Climate Change: Impact Assessment Models",
                "Journal of Sustainable Development: Urban Planning Solutions",
                "Ecology Letters: Biodiversity Conservation Methods",
                "Environmental Research: Pollution Mitigation Approaches"
            ]
        }

        # Determine most likely domain from keywords
        domain_keywords = {
            "medicine": ["health", "patient", "treatment", "disease", "medical", "clinical", "therapy", "diagnostic"],
            "technology": ["algorithm", "computer", "data", "digital", "technology", "network", "software", "computing"],
            "psychology": ["behavior", "cognitive", "mental", "psychology", "brain", "development", "social", "emotional"],
            "environment": ["climate", "environment", "sustainable", "ecological", "conservation", "pollution", "biodiversity", "renewable"]
        }

        # Count keyword matches per domain
        domain_matches = {domain: 0 for domain in domains}
        for keyword in keywords:
            for domain, domain_kws in domain_keywords.items():
                if any(kw in keyword.lower() for kw in domain_kws):
                    domain_matches[domain] += 1

        # Select most matching domain, or random if no matches
        selected_domain = max(domain_matches.items(), key=lambda x: x[1])[0] if any(
            domain_matches.values()) else random.choice(list(domains.keys()))

        results = []
        num_results = random.randint(5, 10)
        for i in range(num_results):
            title = random.choice(domains[selected_domain])
            num_keywords = random.randint(1, len(keywords)) if keywords else 1
            selected_keywords = random.sample(
                keywords, num_keywords) if keywords else []
            domain_specific = random.sample(
                domain_keywords[selected_domain], 2)
            extracted_keywords = ", ".join(selected_keywords + domain_specific)
            relevance = round(0.5 + random.random() * 0.5,
                              2)  # Between 0.5 and 1.0
            result = {
                "title": title,
                "url": f"https://example.com/research/{i+1}",
                "extracted_keywords": extracted_keywords,
                "relevance_score": relevance,
                "source": random.choice(["Google Scholar", "Research Database", "Academic Journal", "Web"])
            }
            results.append(result)

        return results

    def get_optimization_suggestions(self) -> List[str]:
        """
        Generate suggestions for optimizing the system based on historical data.

        Returns:
            List of optimization suggestions
        """
        if not self.historical_metrics:
            return ["Insufficient historical data for optimization suggestions."]

        suggestions = []

        # Identify consistently slow stages
        slow_stages = {}
        for cycle in self.historical_metrics:
            for stage, duration in cycle["stage_durations"].items():
                if stage not in slow_stages:
                    slow_stages[stage] = []
                slow_stages[stage].append(duration)

        # Calculate average durations
        stage_averages = {stage: sum(durations) / len(durations)
                          for stage, durations in slow_stages.items()}

        # Sort stages by average duration (descending)
        sorted_stages = sorted(stage_averages.items(),
                               key=lambda x: x[1], reverse=True)

        # Generate suggestions for the slowest stages
        if sorted_stages:
            slowest_stage, avg_duration = sorted_stages[0]
            if avg_duration > 5:  # If average duration is over 5 seconds
                if "coherence" in slowest_stage.lower():
                    suggestions.append(
                        f"Optimize the {slowest_stage} stage (avg {avg_duration:.1f}s) by simplifying the coherence checks or using a faster model for keyword extraction.")
                elif "generation" in slowest_stage.lower():
                    suggestions.append(
                        f"Optimize the {slowest_stage} stage (avg {avg_duration:.1f}s) by using a smaller, faster model or caching common queries.")
                elif "refinement" in slowest_stage.lower():
                    suggestions.append(
                        f"Optimize the {slowest_stage} stage (avg {avg_duration:.1f}s) by streamlining the refinement prompt or using a more efficient model.")
                else:
                    suggestions.append(
                        f"Optimize the {slowest_stage} stage (avg {avg_duration:.1f}s) which is consistently the slowest part of the process.")

        # Check historical improvement
        if len(self.historical_metrics) >= 3:
            first_cycles = self.historical_metrics[:len(
                self.historical_metrics)//2]
            last_cycles = self.historical_metrics[len(
                self.historical_metrics)//2:]

            first_avg = sum(cycle["total_duration"]
                            for cycle in first_cycles) / len(first_cycles)
            last_avg = sum(cycle["total_duration"]
                           for cycle in last_cycles) / len(last_cycles)

            improvement = ((first_avg - last_avg) / first_avg) * 100

            if improvement < 5:
                suggestions.append(
                    "The system shows minimal performance improvement over time. Consider more fundamental architectural changes.")
            elif improvement > 20:
                suggestions.append(
                    f"The system has improved significantly ({improvement:.1f}%) over time. Continue with the current optimization strategy.")

        # Add general suggestions
        suggestions.append(
            "Consider implementing a caching mechanism for frequently requested queries to reduce processing time.")
        suggestions.append(
            "Evaluate the trade-off between search result quantity and quality to optimize the refinement process.")
        suggestions.append(
            "Web scrapping can be optimised; more data sources needed.")

        return suggestions

    def refine_hypothesis(self, original_query: str, original_hypothesis: str,
                          ranked_keywords: List[Tuple[str, Dict[str, Any]]],
                          search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Refine the hypothesis based on search results and keywords.

        Args:
            original_query: The original user query
            original_hypothesis: The hypothesis from the Generation Agent
            ranked_keywords: Keywords ranked by their relevance
            search_results: Search results to inform the refinement

        Returns:
            Dictionary with refined hypothesis and explanation
        """
        logger.info(f"Refining hypothesis for query: {original_query}")

        prompt = """You are an expert scientific hypothesis refinement system. Your goal is to improve a scientific hypothesis by incorporating:
1. Insights from web search results
2. Keywords ranked by relevance

For your output, provide:
1. A refined, improved hypothesis (2-3 sentences, clearly labeled as "REFINED HYPOTHESIS:")
2. A brief explanation of what was improved (labeled as "EXPLANATION:")
3. A list of 2-3 specific improvements made (labeled as "IMPROVEMENTS:")

Make the hypothesis more precise, evidence-based, and scientifically sound.
"""

        prompt += f"\n\nORIGINAL QUERY: {original_query}"
        prompt += f"\n\nORIGINAL HYPOTHESIS: {original_hypothesis}"

        if ranked_keywords:
            prompt += "\n\nRANKED KEYWORDS (by relevance):"
            for i, (keyword, data) in enumerate(ranked_keywords[:5], 1):
                frequency = data.get("total_frequency", 0)
                boosted = "boosted" in data and data["boosted"]
                boost_indicator = " (boosted)" if boosted else ""
                prompt += f"\n{i}. {keyword} (frequency: {frequency:.1f}){boost_indicator}"

        if search_results:
            prompt += "\n\nTOP SEARCH RESULTS:"
            for i, result in enumerate(search_results[:5], 1):
                title = result.get("title", "")
                extracted_keywords = result.get("extracted_keywords", "")
                prompt += f"\n{i}. {title}"
                if extracted_keywords:
                    prompt += f"\n   Keywords: {extracted_keywords}"

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
            response_text = result["choices"][0]["message"]["content"]

            refined_hypothesis = self._extract_section(
                response_text, "REFINED HYPOTHESIS")
            explanation = self._extract_section(response_text, "EXPLANATION")
            improvements_text = self._extract_section(
                response_text, "IMPROVEMENTS")
            improvements = self._parse_improvements(improvements_text)

            return {
                "original_hypothesis": original_hypothesis,
                "refined_hypothesis": refined_hypothesis if refined_hypothesis else "Unable to generate a refined hypothesis.",
                "explanation": explanation if explanation else "No explanation provided.",
                "improvements": improvements if improvements else ["No specific improvements identified."]
            }

        except Exception as e:
            logger.error(f"Error refining hypothesis: {e}")
            return {
                "original_hypothesis": original_hypothesis,
                "refined_hypothesis": "Error refining hypothesis.",
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
        items = re.findall(
            r'(?:^|\n)(?:\d+\.|\-|\*)\s*(.*?)(?=(?:\n(?:\d+\.|\-|\*))|$)', improvements_text, re.DOTALL)
        if items:
            return [item.strip() for item in items]
        return [line.strip() for line in improvements_text.split('\n') if line.strip()]

    def start_cycle(self) -> Dict[str, Any]:
        """
        Start a new cycle and return timing data structure.

        Returns:
            Timing data dictionary
        """
        timing_data = {
            "cycle_start": time.time(),
            "stages": {},
            "current_stage": None
        }
        return timing_data

    def start_stage(self, timing_data: Dict[str, Any], stage_name: str) -> None:
        """
        Start timing a specific stage.

        Args:
            timing_data: The timing data dictionary
            stage_name: The name of the stage to start
        """
        timing_data["current_stage"] = stage_name
        timing_data["stages"][stage_name] = {
            "start": time.time(),
            "end": None,
            "duration": None
        }

    def end_stage(self, timing_data: Dict[str, Any]) -> float:
        """
        End timing the current stage and return duration.

        Args:
            timing_data: The timing data dictionary

        Returns:
            Duration of the stage in seconds
        """
        stage_name = timing_data["current_stage"]
        if not stage_name or stage_name not in timing_data["stages"]:
            return 0.0

        end_time = time.time()
        timing_data["stages"][stage_name]["end"] = end_time
        start_time = timing_data["stages"][stage_name]["start"]
        duration = end_time - start_time
        timing_data["stages"][stage_name]["duration"] = duration
        timing_data["current_stage"] = None
        return duration

    def end_cycle(self, timing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        End the cycle, calculate total duration and metrics.

        Args:
            timing_data: The timing data dictionary

        Returns:
            Complete timing and metrics data
        """
        if timing_data["current_stage"]:
            self.end_stage(timing_data)

        cycle_end = time.time()
        cycle_duration = cycle_end - timing_data["cycle_start"]
        timing_data["cycle_duration"] = cycle_duration

        cycle_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.process_metrics[cycle_id] = timing_data
        self.historical_metrics.append({
            "cycle_id": cycle_id,
            "total_duration": cycle_duration,
            "stage_durations": {name: stage["duration"] for name, stage in timing_data["stages"].items()}
        })

        return timing_data

    def evaluate_cycle(self,
                       cycle_id: str,
                       query: str,
                       hypothesis: str,
                       coherence_results: Dict[str, Any],
                       ranked_keywords: List[Tuple[str, Dict[str, Any]]],
                       refined_hypothesis: Dict[str, Any],
                       search_results_count: int) -> Dict[str, Any]:
        """
        Evaluate the cycle performance and generate insights.

        Args:
            cycle_id: The ID of the cycle to evaluate
            query: The original query
            hypothesis: The generated hypothesis
            coherence_results: Results from coherence check
            ranked_keywords: Ranked keywords
            refined_hypothesis: The refined hypothesis data
            search_results_count: Number of search results processed

        Returns:
            Evaluation results and insights
        """
        if cycle_id not in self.process_metrics:
            return {
                "status": "error",
                "message": "Cycle not found"
            }

        timing_data = self.process_metrics[cycle_id]
        coherence_score = coherence_results.get("coherence_score", 0)
        keyword_count = len(ranked_keywords) if ranked_keywords else 0
        has_improvements = len(refined_hypothesis.get("improvements", [])) > 0

        stage_durations = [(name, stage["duration"])
                           for name, stage in timing_data["stages"].items()]
        bottlenecks = sorted(
            stage_durations, key=lambda x: x[1], reverse=True)[:2]

        insights = []
        if coherence_score < 5:
            insights.append(
                "The initial hypothesis had low coherence. Consider improving the hypothesis generation prompt.")
        elif coherence_score >= 8:
            insights.append(
                "The initial hypothesis had excellent coherence. The generation agent is performing well.")

        if keyword_count < 3:
            insights.append(
                "Few relevant keywords were extracted. Consider refining the keyword extraction process.")

        if search_results_count < 5:
            insights.append(
                "Few search results were found. Consider broadening the search query or using alternative search methods.")

        for stage_name, duration in bottlenecks:
            if duration > 10:
                insights.append(
                    f"The {stage_name} stage is a performance bottleneck ({duration:.1f} seconds). Consider optimization.")

        if not has_improvements:
            insights.append(
                "No specific improvements were identified in the refinement process. Consider enhancing the Evolution Agent.")

        evaluation = {
            "cycle_id": cycle_id,
            "query": query,
            "performance": {
                "coherence_score": coherence_score,
                "keyword_count": keyword_count,
                "search_results_count": search_results_count,
                "total_duration": timing_data["cycle_duration"],
                "has_improvements": has_improvements
            },
            "bottlenecks": [{"stage": stage, "duration": duration} for stage, duration in bottlenecks],
            "insights": insights
        }

        return evaluation

    def run_complete_cycle(self, query: str = None) -> Dict[str, Any]:
        """
        Run a complete processing cycle for a query.

        Args:
            query: The research query (if None, a random one will be selected)

        Returns:
            Complete cycle results
        """
        if query is None:
            query = random.choice(self.research_topics)

        logger.info(f"Starting complete cycle for query: {query}")
        timing_data = self.start_cycle()
        cycle_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.start_stage(timing_data, "hypothesis_generation")
        hypothesis_data = self.generate_hypothesis(query)
        hypothesis = hypothesis_data["hypothesis"]
        self.end_stage(timing_data)

        self.start_stage(timing_data, "coherence_check")
        coherence_results = self.check_coherence(hypothesis, query)
        self.end_stage(timing_data)

        relevant_keywords = coherence_results.get("relevant_keywords", [])

        self.start_stage(timing_data, "search_generation")
        search_results = self.generate_search_results(relevant_keywords)
        search_results_count = len(search_results)
        self.end_stage(timing_data)

        self.start_stage(timing_data, "keyword_ranking")
        ranked_keywords = self.rank_keywords(relevant_keywords, search_results)
        self.end_stage(timing_data)

        self.start_stage(timing_data, "hypothesis_refinement")
        refined_hypothesis = self.refine_hypothesis(
            query, hypothesis, ranked_keywords, search_results)
        self.end_stage(timing_data)

        cycle_data = self.end_cycle(timing_data)

        evaluation = self.evaluate_cycle(
            cycle_id,
            query,
            hypothesis,
            coherence_results,
            ranked_keywords,
            refined_hypothesis,
            search_results_count
        )

        complete_results = {
            "cycle_id": cycle_id,
            "query": query,
            "hypothesis_generation": {
                "hypothesis": hypothesis,
                "timestamp": hypothesis_data["timestamp"]
            },
            "coherence_check": coherence_results,
            "search_results": {
                "count": search_results_count,
                "results": search_results
            },
            "keyword_ranking": {
                "keywords": ranked_keywords
            },
            "hypothesis_refinement": refined_hypothesis,
            "timing": cycle_data,
            "evaluation": evaluation
        }

        return complete_results

    def run_multiple_cycles(self, topics: List[str] = None, cycles_per_topic: int = 2) -> List[Dict[str, Any]]:
        """
        Run multiple cycles across different topics or multiple cycles per topic.

        Args:
            topics: List of topics to process (defaults to self.research_topics)
            cycles_per_topic: Number of refinement cycles to run per topic

        Returns:
            List of cycle results
        """
        if topics is None:
            topics = self.research_topics

        all_results = []
        for topic in topics:
            logger.info(f"Processing topic: {topic}")
            first_cycle = self.run_complete_cycle(topic)
            all_results.append(first_cycle)
            current_hypothesis = first_cycle["hypothesis_refinement"]["refined_hypothesis"]
            for i in range(1, cycles_per_topic):
                logger.info(
                    f"Starting refinement cycle {i+1} for topic: {topic}")
                next_cycle = self.run_complete_cycle(topic)
                next_cycle["is_refinement"] = True
                next_cycle["previous_hypothesis"] = current_hypothesis
                current_hypothesis = next_cycle["hypothesis_refinement"]["refined_hypothesis"]
                all_results.append(next_cycle)
        return all_results

    def generate_meta_review(self, cycle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a meta-review across all cycles.

        Args:
            cycle_results: List of cycle results

        Returns:
            Meta-review of the system's performance
        """
        if not cycle_results:
            return {
                "status": "error",
                "message": "No cycle results to analyze"
            }

        total_cycles = len(cycle_results)
        avg_coherence = sum(cycle["coherence_check"]["coherence_score"]
                            for cycle in cycle_results) / total_cycles
        avg_keyword_count = sum(len(
            cycle["coherence_check"]["relevant_keywords"]) for cycle in cycle_results) / total_cycles
        avg_duration = sum(cycle["timing"]["cycle_duration"]
                           for cycle in cycle_results) / total_cycles

        all_insights = []
        for cycle in cycle_results:
            cycle_insights = cycle["evaluation"]["insights"]
            all_insights.extend(cycle_insights)

        insight_counts = {}
        for insight in all_insights:
            insight_counts[insight] = insight_counts.get(insight, 0) + 1

        top_insights = sorted(insight_counts.items(),
                              key=lambda x: x[1], reverse=True)[:5]

        optimization_suggestions = self.get_optimization_suggestions()

        refinement_improvements = []
        for cycle in cycle_results:
            improvements = cycle["hypothesis_refinement"].get(
                "improvements", [])
            refinement_improvements.extend(improvements)

        refinement_examples = []
        for i, cycle in enumerate(cycle_results[:3]):
            original = cycle["hypothesis_generation"]["hypothesis"]
            refined = cycle["hypothesis_refinement"]["refined_hypothesis"]
            refinement_examples.append({
                "cycle": i + 1,
                "query": cycle["query"],
                "original": original,
                "refined": refined
            })

        meta_review = {
            "total_cycles": total_cycles,
            "overall_metrics": {
                "average_coherence_score": avg_coherence,
                "average_keyword_count": avg_keyword_count,
                "average_cycle_duration": avg_duration
            },
            "top_insights": [{"insight": insight, "frequency": count} for insight, count in top_insights],
            "optimization_suggestions": optimization_suggestions,
            "refinement_analysis": {
                "common_improvements": list(set(refinement_improvements)),
                "examples": refinement_examples
            }
        }

        return meta_review
