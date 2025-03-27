import logging
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProximityAgent:
    """
    Proximity Agent finds similar past interactions using the vector database
    in the Memory Agent. It provides relevant context from past queries to
    enhance the current interaction cycle.
    """

    def __init__(self, memory_agent):
        """
        Initialize the Proximity Agent with a Memory Agent.

        Args:
            memory_agent: The Memory Agent instance to use for retrieval
        """
        self.memory_agent = memory_agent
        logger.info("Proximity Agent initialized.")

    def find_similar_interactions(self,
                                  query: str,
                                  hypothesis: str = None,
                                  keywords: List[str] = None,
                                  k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar past interactions based on the current query and context.

        Args:
            query: The current user query
            hypothesis: The generated hypothesis (optional)
            keywords: Keywords from the current interaction (optional)
            k: Number of similar interactions to return

        Returns:
            List of similar interactions with similarity scores
        """
        # Enhance the query with hypothesis and keywords if available
        enhanced_query = query
        if hypothesis:
            enhanced_query += f" {hypothesis}"
        if keywords and len(keywords) > 0:
            enhanced_query += f" {' '.join(keywords)}"

        # Get similar interactions from memory
        similar_interactions = self.memory_agent.find_similar(
            enhanced_query, k=k)

        return similar_interactions

    def extract_relevant_information(self, similar_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relevant information from similar interactions to inform the current cycle.

        Args:
            similar_interactions: List of similar interactions from memory

        Returns:
            Dictionary with extracted information
        """
        if not similar_interactions:
            return {
                "similar_queries": [],
                "common_keywords": {},
                "useful_references": [],
                "adaptation_suggestions": [],
                "has_similar_context": False
            }

        # Extract queries from similar interactions
        similar_queries = [
            {
                "query": interaction["query"],
                "similarity": interaction["similarity"],
                "memory_type": interaction.get("memory_type", "unknown")
            }
            for interaction in similar_interactions
        ]

        # Count keyword frequencies
        keyword_frequencies = {}
        for interaction in similar_interactions:
            for keyword_data in interaction.get("ranked_keywords", []):
                keyword = keyword_data.get("keyword")
                if keyword:
                    keyword_frequencies[keyword] = keyword_frequencies.get(
                        keyword, 0) + 1

        # Sort keywords by frequency
        common_keywords = {k: v for k, v in sorted(keyword_frequencies.items(),
                                                   key=lambda item: item[1],
                                                   reverse=True)}

        # Collect useful references from the most similar interaction
        useful_references = []
        if similar_interactions and len(similar_interactions) > 0:
            most_similar = similar_interactions[0]
            for keyword_data in most_similar.get("ranked_keywords", []):
                keyword = keyword_data.get("keyword")
                references = keyword_data.get("references", [])

                for ref in references[:2]:  # Take top 2 references for each keyword
                    useful_references.append({
                        "keyword": keyword,
                        "title": ref.get("title", ""),
                        "url": ref.get("url", ""),
                        "source": "previous search"
                    })

        # Generate adaptation suggestions based on past interactions
        adaptation_suggestions = self._generate_adaptation_suggestions(
            similar_interactions)

        # Determine if there is a strong similarity
        has_similar_context = False
        if similar_interactions and len(similar_interactions) > 0:
            # Consider it similar if similarity > 0.7
            has_similar_context = similar_interactions[0].get(
                "similarity", 0) > 0.7

        return {
            "similar_queries": similar_queries,
            "common_keywords": common_keywords,
            "useful_references": useful_references,
            "adaptation_suggestions": adaptation_suggestions,
            "has_similar_context": has_similar_context
        }

    def _generate_adaptation_suggestions(self, similar_interactions: List[Dict[str, Any]]) -> List[str]:
        """
        Generate suggestions for adaptation based on past interactions.

        Args:
            similar_interactions: List of similar interactions from memory

        Returns:
            List of adaptation suggestions
        """
        suggestions = []

        if not similar_interactions:
            return suggestions

        # Extract keywords that performed well in past interactions
        successful_keywords = []
        for interaction in similar_interactions:
            # Consider keywords from interactions with high coherence as successful
            coherence_score = interaction.get(
                "coherence_results", {}).get("coherence_score", 0)
            if coherence_score >= 7.0:  # Only use keywords from high-scoring hypotheses
                for keyword_data in interaction.get("ranked_keywords", []):
                    # Keywords that had good frequency
                    if keyword_data.get("frequency", 0) > 2:
                        successful_keywords.append(keyword_data.get("keyword"))

        # Create suggestions
        if successful_keywords:
            suggestions.append(
                f"Consider focusing on these keywords that worked well in similar queries: {', '.join(successful_keywords[:3])}")

        # Look for search patterns
        if len(similar_interactions) > 1:
            suggestions.append(
                "Explore diverse sources - previous similar queries benefited from both academic and general web sources")

        # Add suggestion based on most similar query
        if similar_interactions:
            suggestions.append(
                f"The current query is similar to a previous one: '{similar_interactions[0]['query']}'. Consider building upon those findings.")

        # Look for patterns in refined hypotheses
        refinement_patterns = []
        for interaction in similar_interactions:
            agent_outputs = interaction.get("agent_outputs", {})
            if agent_outputs and "refined_hypothesis" in agent_outputs:
                refined_data = agent_outputs["refined_hypothesis"]
                improvements = refined_data.get("improvements", [])
                if improvements:
                    for improvement in improvements:
                        if improvement:
                            refinement_patterns.append(improvement)

        # Add suggestions based on common refinement patterns
        if refinement_patterns:
            suggestions.append(
                f"Previous similar hypotheses were improved by: {refinement_patterns[0]}")

        return suggestions

    def format_output(self, proximity_data: Dict[str, Any], detailed: bool = False) -> str:
        """
        Format the proximity data for display.

        Args:
            proximity_data: The data from extract_relevant_information
            detailed: Whether to include detailed information

        Returns:
            Formatted string with proximity information
        """
        output = "--- PROXIMITY ANALYSIS ---\n"

        # Add similar queries section
        similar_queries = proximity_data.get("similar_queries", [])
        if similar_queries:
            output += "\nSimilar past queries:\n"
            for i, query_data in enumerate(similar_queries, 1):
                memory_type = "recent memory" if query_data["memory_type"] == "working" else "long-term memory"
                similarity = query_data["similarity"] * 100
                output += f"{i}. \"{query_data['query']}\" ({similarity:.1f}% similar, from {memory_type})\n"
        else:
            output += "\nNo similar past queries found.\n"

        # Add common keywords section
        common_keywords = proximity_data.get("common_keywords", {})
        if common_keywords:
            output += "\nCommon keywords from similar interactions:\n"
            top_keywords = list(common_keywords.items())[:5]  # Top 5 keywords
            for keyword, frequency in top_keywords:
                output += f"- {keyword}: appeared in {frequency} similar interactions\n"

        # Add adaptation suggestions
        suggestions = proximity_data.get("adaptation_suggestions", [])
        if suggestions:
            output += "\nSuggestions based on similar past interactions:\n"
            for suggestion in suggestions:
                output += f"- {suggestion}\n"

        # Add useful references if detailed output requested
        if detailed and proximity_data.get("useful_references", []):
            output += "\nUseful references from similar interactions:\n"
            for i, ref in enumerate(proximity_data["useful_references"][:5], 1):
                output += f"{i}. {ref['title']} (keyword: {ref['keyword']})\n"
                output += f"   URL: {ref['url']}\n"

        return output

    def find_topic_related_interactions(self, topic: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find interactions related to a specific topic or keyword.

        Args:
            topic: The topic or keyword to search for
            k: Number of interactions to return

        Returns:
            List of related interactions
        """
        # Get similar interactions from memory using the topic
        related_interactions = self.memory_agent.find_similar(
            topic, k=k, include_long_term=True)

        return related_interactions

    def analyze_research_trends(self, topics: List[str]) -> Dict[str, Any]:
        """
        Analyze research trends across multiple topics.

        Args:
            topics: List of topics to analyze

        Returns:
            Analysis of trends across topics
        """
        results = {}

        for topic in topics:
            # Find interactions related to this topic
            related_interactions = self.find_topic_related_interactions(
                topic, k=3)

            # Extract keywords from these interactions
            topic_keywords = {}
            for interaction in related_interactions:
                for keyword_data in interaction.get("ranked_keywords", []):
                    keyword = keyword_data.get("keyword")
                    if keyword and keyword != topic:  # Exclude the topic itself
                        topic_keywords[keyword] = topic_keywords.get(
                            keyword, 0) + 1

            # Store the top keywords for this topic
            results[topic] = {
                "related_queries": [interaction["query"] for interaction in related_interactions],
                "related_keywords": dict(sorted(topic_keywords.items(), key=lambda x: x[1], reverse=True)[:5])
            }

        # Find cross-topic connections
        connections = {}
        all_keywords = {}

        # Collect all keywords across topics
        for topic, data in results.items():
            for keyword in data["related_keywords"].keys():
                if keyword not in all_keywords:
                    all_keywords[keyword] = []
                all_keywords[keyword].append(topic)

        # Find keywords that appear in multiple topics
        for keyword, topics_list in all_keywords.items():
            if len(topics_list) > 1:
                connections[keyword] = topics_list

        return {
            "topic_analysis": results,
            "cross_topic_connections": connections
        }
