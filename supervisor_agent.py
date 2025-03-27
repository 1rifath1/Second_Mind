import sys
import logging
import json
import datetime
import time
import os
from typing import List, Dict, Any, Optional, Tuple

# Import agents from their respective modules
from generation_agent import GroqTextGenerator, CoherenceChecker
from reflection_agent import ReflectionAgent, groqTextGenerator
from ranking_agent import RankingAgent
from memory_agent import MemoryAgent
from proximity_agent import ProximityAgent
from evolution_agent import EvolutionAgent
from metareview_agent import EnhancedMetaReviewAgent

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedSupervisor:
    """
    Enhanced Supervisor Agent orchestrates all specialized agents:

    1. Generation Agent: Creates initial hypotheses
    2. Reflection Agent: Fetches and enriches search results
    3. Ranking Agent: Scores keywords and outputs
    4. Memory Agent: Stores and retrieves past interactions
    5. Proximity Agent: Finds similar past interactions
    6. Evolution Agent: Refines ideas based on all available data
    7. Meta-Review Agent: Evaluates the process and suggests improvements

    The Supervisor manages the flow between agents, allocates resources dynamically,
    and enables feedback loops for continuous improvement.
    """

    def __init__(self,
                 serapi_key="68f8b34c3a5ab43ed7a061dfb3ae6869c42a5996048688a065be25824d9b5e70",
                 groq_api_key="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",
                 pinecone_api_key="pcsk_2yFRRb_GZQShHjGazfM6TTSdLGKDA4iUSaiRkB3yKt7q6mvUHDFwkZkLiJuYWQ6uCdM7Vk"):
        """
        Initialize the Enhanced Supervisor with all specialized agents.

        Args:
            serapi_key: SerpAPI key for web searches
            groq_api_key: Groq API key for text generation
            pinecone_api_key: Pinecone API key for vector database
        """
        self.serapi_key = serapi_key
        self.groq_api_key = groq_api_key
        self.pinecone_api_key = pinecone_api_key

        # Initialize all agents
        logger.info("Initializing specialized agents...")

        # Generation agents
        self.text_generator = GroqTextGenerator(api_key=self.groq_api_key)
        self.coherence_checker = CoherenceChecker(api_key=self.groq_api_key)

        # Reflection agent
        self.reflection_agent = ReflectionAgent(serapi_key=self.serapi_key)
        self.reflection_text_generator = groqTextGenerator(
            api_key=self.groq_api_key)

        # Ranking agent
        self.ranking_agent = RankingAgent()

        # Memory and Proximity agents
        self.memory_agent = MemoryAgent(api_key=self.pinecone_api_key)
        self.proximity_agent = ProximityAgent(self.memory_agent)

        # Evolution agent
        self.evolution_agent = EvolutionAgent(api_key=self.groq_api_key)

        # Meta-review agent
        self.meta_review_agent = EnhancedMetaReviewAgent()

        # Create results directory
        os.makedirs("results", exist_ok=True)

        logger.info(
            "Enhanced Supervisor Agent initialized with all specialized agents.")

    def run_cycle(self, prompt: str, detailed_output: bool = False, enable_memory: bool = True) -> Dict[str, Any]:
        """
        Run a complete cycle of the Second Mind system with all agents.

        Args:
            prompt: The user's query or prompt
            detailed_output: Whether to include detailed output
            enable_memory: Whether to use memory features

        Returns:
            Complete results from the cycle
        """
        # Start Meta-Review timing
        timing_data = self.meta_review_agent.start_cycle()
        cycle_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "cycle_id": cycle_id,
            "prompt": prompt,
            "timestamp": datetime.datetime.now().isoformat(),
            "stages": {}
        }

        try:
            # STAGE 1: Check for similar past interactions with Proximity Agent
            if enable_memory:
                self.meta_review_agent.start_stage(
                    timing_data, "proximity_search")
                logger.info("Checking for similar past interactions...")

                similar_interactions = self.proximity_agent.find_similar_interactions(
                    prompt)
                proximity_data = self.proximity_agent.extract_relevant_information(
                    similar_interactions)

                results["stages"]["proximity"] = {
                    "similar_interactions": similar_interactions,
                    "proximity_data": proximity_data
                }

                # Log proximity results
                if similar_interactions:
                    logger.info(
                        f"Found {len(similar_interactions)} similar past interactions")
                else:
                    logger.info("No similar past interactions found")

                self.meta_review_agent.end_stage(timing_data)
            else:
                proximity_data = None

            # STAGE 2: Generate hypothesis with Generation Agent
            self.meta_review_agent.start_stage(
                timing_data, "hypothesis_generation")
            logger.info("Generating hypothesis...")

            hypothesis = self.text_generator.generate_text(prompt)
            coherence_results = self.coherence_checker.check_coherence(
                hypothesis, prompt)

            results["stages"]["generation"] = {
                "hypothesis": hypothesis,
                "coherence_results": coherence_results
            }

            logger.info(
                f"Generated hypothesis with coherence score: {coherence_results.get('coherence_score', 0)}/10")

            self.meta_review_agent.end_stage(timing_data)

            # STAGE 3: Fetch search results with Reflection Agent
            self.meta_review_agent.start_stage(
                timing_data, "search_reflection")
            logger.info("Fetching and analyzing search results...")

            search_results = self.reflection_agent.analyze_search_results(
                query=prompt)
            logger.info(f"Fetched {len(search_results)} search results")

            # Enrich results with keywords
            logger.info("Enriching search results with keywords...")
            self.reflection_agent.enrich_results_with_keywords(
                hypothesis, self.reflection_text_generator)

            results["stages"]["reflection"] = {
                # Limit unless detailed output requested
                "search_results": search_results[:5] if not detailed_output else search_results,
                "extracted_keywords": self.reflection_agent.extracted_keywords_list
            }

            self.meta_review_agent.end_stage(timing_data)

            # Get hypothesis keywords from coherence check
            hypothesis_keywords = coherence_results.get(
                "relevant_keywords", [])

            # STAGE 4: Rank keywords with Ranking Agent
            self.meta_review_agent.start_stage(timing_data, "keyword_ranking")
            logger.info("Ranking keywords...")

            ranked_keywords = self.ranking_agent.rank_keywords(
                hypothesis_keywords, self.reflection_agent.last_search_results)

            results["stages"]["ranking"] = {
                "ranked_keywords": ranked_keywords
            }

            logger.info(f"Ranked {len(ranked_keywords)} keywords")

            self.meta_review_agent.end_stage(timing_data)

            # STAGE 5: Refine hypothesis with Evolution Agent
            self.meta_review_agent.start_stage(
                timing_data, "hypothesis_evolution")
            logger.info("Refining hypothesis...")

            refined_hypothesis = self.evolution_agent.refine_hypothesis(
                original_query=prompt,
                original_hypothesis=hypothesis,
                ranked_keywords=ranked_keywords,
                search_results=self.reflection_agent.last_search_results,
                proximity_data=proximity_data
            )

            results["stages"]["evolution"] = {
                "refined_hypothesis": refined_hypothesis
            }

            logger.info("Hypothesis refined successfully")

            self.meta_review_agent.end_stage(timing_data)

            # STAGE 6: Store in Memory if enabled
            if enable_memory:
                self.meta_review_agent.start_stage(
                    timing_data, "memory_storage")
                logger.info("Storing interaction in memory...")

                memory_id = self.memory_agent.store_interaction(
                    query=prompt,
                    hypothesis=hypothesis,
                    coherence_results=coherence_results,
                    ranked_keywords=ranked_keywords,
                    search_results=self.reflection_agent.last_search_results,
                    agent_outputs={
                        "refined_hypothesis": refined_hypothesis,
                        "proximity_data": proximity_data
                    }
                )

                results["stages"]["memory"] = {
                    "memory_id": memory_id
                }

                logger.info(
                    f"Interaction stored in memory with ID: {memory_id}")

                self.meta_review_agent.end_stage(timing_data)

            # STAGE 7: Meta-Review evaluation
            self.meta_review_agent.start_stage(timing_data, "meta_review")
            logger.info("Performing meta-review evaluation...")

            # End cycle and get complete timing data
            timing_results = self.meta_review_agent.end_cycle(timing_data)

            # Evaluate the cycle
            evaluation = self.meta_review_agent.evaluate_cycle(
                cycle_id=cycle_id,
                query=prompt,
                hypothesis=hypothesis,
                coherence_results=coherence_results,
                ranked_keywords=ranked_keywords,
                refined_hypothesis=refined_hypothesis,
                search_results_count=len(
                    self.reflection_agent.last_search_results)
            )

            # Get optimization suggestions
            optimization_suggestions = self.meta_review_agent.get_optimization_suggestions()

            results["stages"]["meta_review"] = {
                "timing": timing_results,
                "evaluation": evaluation,
                "optimization_suggestions": optimization_suggestions
            }

            logger.info("Meta-review completed")

            self.meta_review_agent.end_stage(timing_data)

            # Save results
            self._save_results(results, cycle_id)

            return results

        except Exception as e:
            logger.error(f"Error during cycle execution: {e}")
            # End current stage if there is one
            if timing_data["current_stage"]:
                self.meta_review_agent.end_stage(timing_data)

            # End cycle
            self.meta_review_agent.end_cycle(timing_data)

            return {
                "cycle_id": cycle_id,
                "status": "error",
                "error": str(e),
                "prompt": prompt,
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _save_results(self, results: Dict[str, Any], cycle_id: str) -> str:
        """
        Save cycle results to a file.

        Args:
            results: The results to save
            cycle_id: The cycle ID

        Returns:
            Path to the saved file
        """
        # Create full results directory
        results_dir = os.path.join("results", cycle_id)
        os.makedirs(results_dir, exist_ok=True)

        # Save complete results
        results_path = os.path.join(results_dir, "complete_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary = self._generate_summary(results)
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {results_path}")
        return results_path

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the results.

        Args:
            results: The complete results

        Returns:
            Summary dictionary
        """
        return {
            "cycle_id": results.get("cycle_id", ""),
            "prompt": results.get("prompt", ""),
            "timestamp": results.get("timestamp", ""),
            "hypothesis": results.get("stages", {}).get("generation", {}).get("hypothesis", ""),
            "coherence_score": results.get("stages", {}).get("generation", {}).get("coherence_results", {}).get("coherence_score", 0),
            "refined_hypothesis": results.get("stages", {}).get("evolution", {}).get("refined_hypothesis", {}).get("refined_hypothesis", ""),
            "top_keywords": [kw for kw, _ in results.get("stages", {}).get("ranking", {}).get("ranked_keywords", [])[:5]],
            "similar_interactions_count": len(results.get("stages", {}).get("proximity", {}).get("similar_interactions", [])),
            "cycle_duration": results.get("stages", {}).get("meta_review", {}).get("timing", {}).get("cycle_duration", 0)
        }

    def run_interactive(self):
        """
        Run the Second Mind system in interactive mode.
        """
        print("=== THE SECOND MIND - INTERACTIVE MODE ===")
        print("Welcome to The Second Mind system! Enter your queries below.")
        print("Type 'exit' or 'quit' to exit the system.\n")

        memory_enabled = True
        detailed_output = False

        while True:
            prompt = input("\nEnter your query: ").strip()

            if prompt.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not prompt:
                print("Please enter a valid query.")
                continue

            # Check for special commands
            if prompt.startswith("/"):
                if prompt == "/memory on":
                    memory_enabled = True
                    print("Memory features enabled.")
                    continue
                elif prompt == "/memory off":
                    memory_enabled = False
                    print("Memory features disabled.")
                    continue
                elif prompt == "/detail on":
                    detailed_output = True
                    print("Detailed output enabled.")
                    continue
                elif prompt == "/detail off":
                    detailed_output = False
                    print("Detailed output disabled.")
                    continue
                elif prompt == "/help":
                    self._print_help()
                    continue
                elif prompt == "/stats":
                    self._print_stats()
                    continue

            # Run cycle
            print(f"\nProcessing query: '{prompt}'")
            print("This may take a moment as multiple agents are working together...\n")

            start_time = time.time()
            results = self.run_cycle(prompt, detailed_output, memory_enabled)
            duration = time.time() - start_time

            # Display results
            self._display_results(results, detailed_output)
            print(f"\nTotal processing time: {duration:.2f} seconds")

    def _print_help(self):
        """Display help information."""
        print("\n=== THE SECOND MIND - HELP ===")
        print("Available commands:")
        print("  /memory on    - Enable memory features")
        print("  /memory off   - Disable memory features")
        print("  /detail on    - Enable detailed output")
        print("  /detail off   - Disable detailed output")
        print("  /stats        - Show system statistics")
        print("  /help         - Show this help message")
        print("  exit, quit, q - Exit the system")
        print("\nRegular usage:")
        print("  Simply type your query and press Enter")
        print("  Example: Renewable energy solutions for urban environments")

    def _print_stats(self):
        """Display system statistics."""
        print("\n=== THE SECOND MIND - STATISTICS ===")

        # Memory stats
        memory_keywords = self.memory_agent.get_all_keywords()
        keyword_count = len(memory_keywords)

        # Get total memory entries
        memory_entry_count = len(self.memory_agent.memory_entries)

        # Meta-review stats
        cycle_count = len(self.meta_review_agent.historical_metrics)

        if cycle_count > 0:
            avg_duration = sum(metric["total_duration"]
                               for metric in self.meta_review_agent.historical_metrics) / cycle_count

            # Get common stages
            stage_durations = {}
            for metric in self.meta_review_agent.historical_metrics:
                for stage, duration in metric["stage_durations"].items():
                    if stage not in stage_durations:
                        stage_durations[stage] = []
                    stage_durations[stage].append(duration)

            # Calculate averages
            stage_averages = {stage: sum(durations) / len(durations)
                              for stage, durations in stage_durations.items()}

            # Print stats
            print(f"Total cycles completed: {cycle_count}")
            print(f"Average cycle duration: {avg_duration:.2f} seconds")
            print("\nAverage stage durations:")
            for stage, avg in sorted(stage_averages.items(), key=lambda x: x[1], reverse=True):
                print(f"  {stage}: {avg:.2f} seconds")
        else:
            print("No cycles completed yet.")

        print(f"\nMemory stats:")
        print(f"  Total interactions stored: {memory_entry_count}")
        print(f"  Unique keywords: {keyword_count}")

        if keyword_count > 0:
            print("\nTop keywords in memory:")
            top_keywords = sorted(memory_keywords.items(),
                                  key=lambda x: x[1], reverse=True)[:5]
            for keyword, count in top_keywords:
                print(f"  {keyword}: {count} occurrences")

    def _display_results(self, results: Dict[str, Any], detailed: bool = False):
        """
        Display results to the user.

        Args:
            results: The results to display
            detailed: Whether to show detailed output
        """
        if "error" in results:
            print(f"Error: {results['error']}")
            return

        # Display initial hypothesis
        hypothesis = results.get("stages", {}).get(
            "generation", {}).get("hypothesis", "")
        coherence_results = results.get("stages", {}).get(
            "generation", {}).get("coherence_results", {})
        coherence_score = coherence_results.get("coherence_score", 0)
        coherence_verdict = coherence_results.get("verdict", "")

        print("\n=== INITIAL HYPOTHESIS ===")
        print(hypothesis)
        print(f"Coherence: {coherence_score}/10 - {coherence_verdict}")

        # Display keywords
        keywords = coherence_results.get("relevant_keywords", [])
        if keywords:
            print(f"Keywords: {', '.join(keywords)}")

        # Display proximity results if available
        proximity_data = results.get("stages", {}).get(
            "proximity", {}).get("proximity_data", {})
        if proximity_data and proximity_data.get("similar_queries", []):
            print("\n=== SIMILAR PAST QUERIES ===")
            for i, query_data in enumerate(proximity_data.get("similar_queries", [])[:3], 1):
                similarity = query_data.get("similarity", 0) * 100
                print(
                    f"{i}. \"{query_data['query']}\" ({similarity:.1f}% similar)")

        # Display ranked keywords
        ranked_keywords = results.get("stages", {}).get(
            "ranking", {}).get("ranked_keywords", [])
        if ranked_keywords:
            print("\n=== RANKED KEYWORDS ===")
            for i, (keyword, data) in enumerate(ranked_keywords[:5], 1):
                frequency = data.get("total_frequency", 0)
                print(f"{i}. {keyword} (frequency: {frequency})")

        # Display refined hypothesis - FIX THE DISPLAY ISSUE HERE
        evolution_data = results.get("stages", {}).get("evolution", {})
        refined_data = evolution_data.get(
            "refined_hypothesis", {}) if evolution_data else {}

        # Safely extract data with explicit checks
        refined_hypothesis = refined_data.get(
            "refined_hypothesis", "") if refined_data else ""
        explanation = refined_data.get(
            "explanation", "") if refined_data else ""
        improvements = refined_data.get(
            "improvements", []) if refined_data else []

        print("\n=== REFINED HYPOTHESIS ===")
        if refined_hypothesis:
            print(refined_hypothesis)
        else:
            print("No refined hypothesis was generated.")

        if explanation:
            print("\nExplanation of improvements:")
            print(explanation)

        if improvements:
            print("\nSpecific improvements made:")
            for i, improvement in enumerate(improvements, 1):
                print(f"{i}. {improvement}")
        elif refined_hypothesis:
            print("\nNo specific improvements were identified.")

        # Display meta-review insights if detailed output requested
        if detailed:
            evaluation = results.get("stages", {}).get(
                "meta_review", {}).get("evaluation", {})
            insights = evaluation.get("insights", [])

            if insights:
                print("\n=== META-REVIEW INSIGHTS ===")
                for insight in insights:
                    print(f"- {insight}")

            # Display search results sample
            search_results = results.get("stages", {}).get(
                "reflection", {}).get("search_results", [])
            if search_results:
                print("\n=== SEARCH RESULTS SAMPLE ===")
                for i, result in enumerate(search_results[:3], 1):
                    print(f"{i}. {result.get('title', '')}")
                    print(f"   URL: {result.get('url', '')}")
                    if result.get("extracted_keywords"):
                        print(
                            f"   Keywords: {result.get('extracted_keywords', '')}")

        # Display cycle ID for reference
        print(f"\nCycle ID: {results.get('cycle_id', '')}")


def main():
    """Main function to run the Enhanced Supervisor."""
    print("Initializing The Second Mind system...")

    # Check if API keys are available
    serapi_key = os.environ.get(
        "SERAPI_KEY", "68f8b34c3a5ab43ed7a061dfb3ae6869c42a5996048688a065be25824d9b5e70")
    groq_api_key = os.environ.get(
        "GROQ_API_KEY", "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")
    pinecone_api_key = os.environ.get(
        "PINECONE_API_KEY", "pcsk_2yFRRb_GZQShHjGazfM6TTSdLGKDA4iUSaiRkB3yKt7q6mvUHDFwkZkLiJuYWQ6uCdM7Vk")

    # Initialize the supervisor
    supervisor = EnhancedSupervisor(
        serapi_key=serapi_key,
        groq_api_key=groq_api_key,
        pinecone_api_key=pinecone_api_key
    )

    # Run in interactive mode
    supervisor.run_interactive()


if __name__ == "__main__":
    main()
