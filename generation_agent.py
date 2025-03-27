import os
import requests
import json
import random
import datetime
import re

# ---------------------- Generation Section ---------------------- #
class GroqTextGenerator:
    def __init__(self, api_key="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"):
        """
        Initialize the Groq text generator with an API key.
        
        Args:
            api_key (str): Groq API key.
        """
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt, model="llama3-70b-8192", temperature=0.7, max_tokens=200):
        """
        Generate text using the Groq API.
        
        Args:
            prompt (str): The prompt to generate text from.
            model (str): The model to use for generation.
            temperature (float): Controls randomness (0.0-1.0).
            max_tokens (int): Maximum number of tokens to generate.
            
        Returns:
            str: The generated text.
        """
        # Add randomization to ensure different responses
        adjusted_temp = min(1.0, max(0.1, temperature + random.uniform(-0.1, 0.1)))
        random_seed = random.randint(1, 10000)
        
        # Format the prompt to request a concise scientific hypothesis
        formatted_prompt = f"Generate a concise scientific hypothesis about: {prompt}. Keep it under 2-3 sentences."
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": adjusted_temp,
            "max_tokens": max_tokens,
            "seed": random_seed
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Groq API: {str(e)}"
            if hasattr(e, 'response') and e.response:
                try:
                    error_details = e.response.json()
                    error_msg += f"\nDetails: {json.dumps(error_details, indent=2)}"
                except:
                    error_msg += f"\nStatus code: {e.response.status_code}"
            return f"Error: {error_msg}"


# ---------------------- Reflection Section ---------------------- #
class CoherenceChecker:
    """
    A simplified coherence checker that only returns the overall coherence score (with verdict)
    and up to the 5 most relevant keywords extracted from the hypothesis.
    """
    def __init__(self, api_key="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"):
        """
        Initialize the coherence checker.
        
        Args:
            api_key (str): API key for the LLM to enable keyword extraction.
        """
        self.api_key = api_key

    def check_coherence(self, hypothesis, prompt):
        """
        Check the coherence of a hypothesis and extract up to 5 relevant keywords.
        
        Args:
            hypothesis (str): The hypothesis to check.
            prompt (str): The original prompt.
            
        Returns:
            dict: A dictionary with overall coherence score, verdict, and relevant keywords.
        """
        sentences = re.split(r'(?<=[.!?])\s+', hypothesis)
        
        # Calculate various factors used to determine coherence.
        structure = {
            "has_hypothesis_structure": self._check_hypothesis_structure(hypothesis),
            "has_causal_language": self._check_causal_language(hypothesis),
            "has_scientific_terms": self._check_scientific_terms(hypothesis)
        }
        grammar_issues = self._check_grammar(hypothesis, sentences)
        relevance = {
            "relevant_to_prompt": self._check_relevance(hypothesis, prompt),
            "relevant_keywords": self._extract_relevant_keywords(hypothesis, prompt)
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
        coherence_score = round(sum(1 for factor in coherence_factors if factor) / len(coherence_factors) * 10, 1)
        
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
            "relevant_keywords": relevance["relevant_keywords"]
        }
    
    def _check_hypothesis_structure(self, text):
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
    
    def _check_causal_language(self, text):
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
    
    def _check_scientific_terms(self, text):
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
    
    def _check_grammar(self, text, sentences):
        """Perform basic grammar checks."""
        issues = []
        # Check for repeated words
        repeated = re.findall(r'\b(\w+)\s+\1\b', text.lower())
        if repeated:
            issues.append(f"Repeated words: {', '.join(repeated)}")
        # Check for overly long sentences (more than 40 words)
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) > 40:
                issues.append(f"Sentence {i+1} is very long ({len(sentence.split())} words)")
        # Check if each sentence starts with a capital letter
        for i, sentence in enumerate(sentences):
            if sentence and not sentence[0].isupper():
                issues.append(f"Sentence {i+1} does not begin with a capital letter")
        return issues
    
    def _check_relevance(self, hypothesis, prompt):
        """Check if the hypothesis is relevant to the prompt."""
        if prompt.lower() in hypothesis.lower():
            return True
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
        hypothesis_words = set(re.findall(r'\b\w{4,}\b', hypothesis.lower()))
        return len(prompt_words.intersection(hypothesis_words)) >= 1
    
    def _extract_relevant_keywords(self, hypothesis, prompt):
        """
        Use an LLM to extract a comma-separated list of keywords that capture 
        the connection between the hypothesis and the prompt, limited to 5 keywords.
        """
        keywords_str = self._llm_generate_keywords(hypothesis, prompt)
        if keywords_str:
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
            return keywords[:5]
        return []
    
    def _llm_generate_keywords(self, hypothesis, prompt):
        """
        Call the LLM to generate keywords.
        
        Constructs a prompt asking the LLM to extract a comma-separated list of keywords.
        
        Returns:
            str: The keywords string returned by the LLM.
        """
        if not self.api_key:
            return ""
        
        base_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        model = "llama3-70b-8192"
        system_message = (
            "You are an expert text analyzer. Given a hypothesis and its prompt, "
            "extract a comma-separated list of relevant keywords that highlight the connection between them. "
            "Only output the keywords, with no extra commentary."
        )
        user_message = f"Hypothesis: {hypothesis}\nPrompt: {prompt}\nKeywords:"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.3,
            "max_tokens": 60
        }
        try:
            response = requests.post(base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            keywords_output = result["choices"][0]["message"]["content"]
            return keywords_output
        except Exception as e:
            print(f"Error calling LLM for keyword extraction: {e}")
            return ""
    
    def display_results(self, results):
        """Display the overall coherence score and relevant keywords."""
        print("\nCOHERENCE ANALYSIS:")
        print(f"Overall score: {results['coherence_score']}/10 - {results['verdict']}")
        if results['relevant_keywords']:
            print(f"Relevant keywords: {', '.join(results['relevant_keywords'])}")
        else:
            print("No relevant keywords extracted.")


def save_to_json(prompt, hypothesis, coherence_results):
    """Save the prompt, hypothesis, and coherence analysis results to a JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coherence_analysis_{timestamp}.json"
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "hypothesis": hypothesis,
        "coherence_analysis": coherence_results
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return filename


# ---------------------- Main Execution Flow ---------------------- #
def main():
    # Use the same API key for text generation.
    generator = GroqTextGenerator(api_key="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")
    
    # Get the topic for hypothesis generation from the user.
    prompt = input("Enter a topic for hypothesis generation: ")
    print("\nGenerating a hypothesis for the prompt...\n")
    hypothesis = generator.generate_text(prompt)
    print("Generated hypothesis:\n")
    print(hypothesis)
    
    # Use a separate API key for LLM keyword extraction if desired.
    api_key_llm = input("\nEnter API key for LLM keyword extraction (or leave blank to use the same key): ").strip() or "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"
    checker = CoherenceChecker(api_key=api_key_llm)
    coherence_results = checker.check_coherence(hypothesis, prompt)
    checker.display_results(coherence_results)
    
    # Optionally save the coherence analysis.
    save_choice = input("\nDo you want to save this coherence analysis? (y/n): ").strip().lower()
    if save_choice == "y":
        filename = save_to_json(prompt, hypothesis, coherence_results)
        print(f"Analysis saved to {filename}")

if __name__ == "__main__":
    main()
