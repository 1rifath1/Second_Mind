import random


class RankingAgent:
    """
    Ranking Agent with forced cycling and enhanced keyword processing
    """

    def __init__(self):
        self.cycle_count = 0
        self.previous_selected_keywords = []

    def rank_keywords(self, hypothesis_keywords, results):
        """
        Rank keywords with a forced cycling mechanism

        Args:
            hypothesis_keywords (list): Original keywords from the hypothesis
            results (list): Search results to rank against

        Returns:
            list: Ranked and potentially cycled keywords
        """
        # Standard ranking process
        ranking = {}
        for keyword in hypothesis_keywords:
            kw_lower = keyword.lower()
            total_frequency = 0
            references = []
            for result in results:
                extracted = result.get("extracted_keywords", "")
                count = extracted.lower().count(kw_lower)
                if count > 0:
                    total_frequency += count
                    references.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "count": count,
                        "source": result.get("source", "Unknown")
                    })
            ranking[keyword] = {
                "total_frequency": total_frequency,
                "references": references
            }

        # Sort keywords by total frequency (highest first)
        sorted_keywords = sorted(
            ranking.items(), key=lambda x: x[1]["total_frequency"], reverse=True)

        # Forced cycling mechanism
        self.cycle_count += 1

        # If we have at least 4 keywords, use a cycling strategy
        if len(sorted_keywords) >= 4:
            # Select ranks 2, 3, and 4 (0-indexed, so 1, 2, 3)
            cycle_candidates = sorted_keywords[1:4]

            # Randomly select one keyword to be promoted
            if cycle_candidates:
                selected_keyword = random.choice(cycle_candidates)

                # Artificially boost the selected keyword's rank
                selected_keyword[1]["total_frequency"] *= 1.5

                # Ensure we don't repeatedly select the same keyword
                while selected_keyword[0] in self.previous_selected_keywords:
                    selected_keyword = random.choice(cycle_candidates)

                # Update the previous selections
                self.previous_selected_keywords.append(selected_keyword[0])
                if len(self.previous_selected_keywords) > 3:
                    self.previous_selected_keywords.pop(0)

        # Re-sort after potential modifications
        sorted_keywords = sorted(
            ranking.items(), key=lambda x: x[1]["total_frequency"], reverse=True)

        return sorted_keywords

    def get_comprehensive_keyword_details(self, ranked_keywords):
        """
        Generate comprehensive details for the top keywords

        Args:
            ranked_keywords (list): Ranked keywords from rank_keywords method

        Returns:
            dict: Detailed information about top keywords
        """
        keyword_details = {}
        for keyword, data in ranked_keywords[:3]:  # Top 3 keywords
            # Separate Scholar and Web references
            scholar_refs = [
                ref for ref in data['references']
                if ref.get('source', '').lower() in ['google scholar', 'academic', 'research']
            ]
            web_refs = [
                ref for ref in data['references']
                if ref.get('source', '').lower() not in ['google scholar', 'academic', 'research']
            ]

            keyword_details[keyword] = {
                "total_frequency": data['total_frequency'],
                # Top 3 scholar references
                "scholar_references": scholar_refs[:3],
                "web_references": web_refs[:3],  # Top 3 web references
                "cycle_boost": self.cycle_count  # Indicate if this keyword was artificially boosted
            }

        return keyword_details
