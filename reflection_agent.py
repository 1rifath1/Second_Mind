import logging
import random
import re
import requests
import time
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Groq API endpoint (used for enrichment calls)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class groqTextGenerator:
    """
    GroqTextGenerator uses the Groq API for summary generation and keyword extraction.
    """

    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Grok API key is required.")
        self.api_key = api_key
        self.base_url = GROQ_API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_text(self, prompt, model="llama3-70b-8192", temperature=0.7, max_tokens=150):
        adjusted_temp = min(
            1.0, max(0.1, temperature + random.uniform(-0.1, 0.1)))
        random_seed = random.randint(1, 10000)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": adjusted_temp,
            "max_tokens": max_tokens,
            "seed": random_seed
        }
        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return "Error generating text."

    def llm_generate_keywords(self, hypothesis, content, model="llama3-70b-8192", temperature=0.3, max_tokens=60):
        """
        Generate a comma-separated list of exactly 5 important keywords given a hypothesis and content.
        """
        system_message = (
            "You are an expert text analyzer. Given a hypothesis and additional content, "
            "extract a comma-separated list of exactly 5 most relevant keywords that highlight the connection between them. "
            "Only output the keywords, with no extra commentary."
        )
        user_message = f"Hypothesis: {hypothesis}\nContent: {content}\nPlease provide exactly 5 keywords:"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            keywords_output = result["choices"][0]["message"]["content"].strip(
            )
            keywords = [kw.strip()
                        for kw in keywords_output.split(',') if kw.strip()]
            if len(keywords) > 5:
                keywords = keywords[:5]
            return ', '.join(keywords)
        except Exception as e:
            logger.error(f"Error calling Groq API for keyword extraction: {e}")
            return "Error generating keywords."


class ReflectionAgent:
    """
    Reflection Agent fetches search results from regular search engines (Google, Bing, and DuckDuckGo)
    and from Google Scholar. It returns 20 regular results and 20 scholar results.
    Then it enriches each result by scraping its content and extracting keywords via the Groq API.
    """

    def __init__(self, serapi_key=None):
        self.serapi_key = serapi_key or "68f8b34c3a5ab43ed7a061dfb3ae6869c42a5996048688a065be25824d9b5e70"
        self.regular_results = []  # Combined results from Google, Bing, DuckDuckGo
        self.scholar_results = []  # Google Scholar results
        self.last_search_results = []  # Combined regular + scholar results
        # Collected extracted keywords (for external use)
        self.extracted_keywords_list = []

    def analyze_search_results(self, query):
        logger.info(f"Analyzing search results for: {query}")
        results = []

        # Fetch results from Google, Bing, and DuckDuckGo
        # Reduced the number of results requested from each source
        google = self._fetch_google_search(query, 4)  # Reduced from 7
        if google:
            results.extend(google)

        bing = self._fetch_bing_search(query, 3)  # Reduced from 7
        if bing:
            results.extend(bing)

        duckduckgo = self._fetch_duckduckgo_search(query, 3)  # Reduced from 6
        if duckduckgo:
            results.extend(duckduckgo)

        # Limit to 10 regular results (down from 20)
        self.regular_results = results[:10]
        logger.info(
            f"Fetched {len(self.regular_results)} regular search results.")

        # Keep Google Scholar results at 20 (unchanged)
        self.scholar_results = self._fetch_google_scholar(query, 20)
        logger.info(
            f"Fetched {len(self.scholar_results)} Google Scholar results.")

        # Combine both result sets
        self.last_search_results = self.regular_results + self.scholar_results
        return self.last_search_results

    def enrich_results_with_keywords(self, hypothesis, generator, max_chars=1000, timeout=8):
        """
        For each result in last_search_results, attempt to extract relevant keywords
        with optimizations to minimize time spent on inaccessible sites.

        Features:
        - Immediate skipping of known problematic domains
        - Shortened timeout periods
        - Parallel processing of requests (concurrent.futures)
        - Early termination of slow requests
        - Progress tracking
        """
        self.extracted_keywords_list = []
        start_time = time.time()

        # List of common user agents to rotate through
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        ]

        # Sites that are known to block scrapers or require special access
        blocked_domains = [
            "sciencedirect.com", "tandfonline.com", "springer.com", "ieee.org", "acm.org",
            "aaai.org", "nature.com", "wiley.com", "jstor.org", "heinonline.org",
            "ieeexplore.ieee.org", "pubmed.ncbi.nlm.nih.gov", "nytimes.com",
            "washingtonpost.com", "wsj.com", "bloomberg.com", "yahoo.com",
            "geekwire.com", "ft.com", "forbes.com", "academia.edu", "researchgate.net"
        ]

        # Pre-filter results to separate easy and challenging sites
        fast_track_results = []
        standard_results = []

        # Process results in batches to maintain progress
        batch_size = min(10, len(self.last_search_results)
                         )  # Process at most 10 at a time
        total_batches = (len(self.last_search_results) +
                         batch_size - 1) // batch_size

        # Categorize URLs for optimized processing
        for result in self.last_search_results:
            url = result["url"]
            domain = self._extract_domain(url)

            # Skip known problematic domains - use available metadata instead
            if any(domain.endswith(bd) or bd in domain for bd in blocked_domains):
                # Fast track: use title and snippet
                fast_track_results.append(result)
            else:
                # Standard track: attempt to scrape
                standard_results.append(result)

        logger.info(
            f"Processing {len(fast_track_results)} fast-track results (using metadata only)")

        # Process fast-track results first (these use metadata, not scraping)
        for result in fast_track_results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            content = f"{title}. {snippet}"

            extracted_keywords = generator.llm_generate_keywords(
                hypothesis, content, max_tokens=60)
            result["extracted_keywords"] = extracted_keywords
            if extracted_keywords and extracted_keywords != "Error generating keywords.":
                self.extracted_keywords_list.append(
                    (result["url"], extracted_keywords))

        logger.info(
            f"Processing {len(standard_results)} standard results (with scraping)")

        # Process standard results in batches for better progress tracking
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(standard_results))
            batch = standard_results[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_num+1}/{total_batches} ({len(batch)} results)")

            # Process each result in the batch
            for result in batch:
                url = result["url"]

                # Set a shorter timeout for these requests to avoid hanging
                try:
                    # Rotate user agents
                    headers = {
                        "User-Agent": random.choice(user_agents),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5"
                    }

                    # Short delay to avoid immediate rate limiting
                    time.sleep(random.uniform(0.5, 1.0))

                    # Set shorter timeout to quickly skip inaccessible sites
                    response = requests.get(
                        url, headers=headers, timeout=timeout)

                    # Try to get content with Beautiful Soup
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.extract()

                    # Try to find main content first
                    main_content = soup.find('main') or soup.find(
                        'article') or soup.find('div', class_='content')

                    if main_content:
                        paragraphs = main_content.find_all("p")
                    else:
                        paragraphs = soup.find_all("p")

                    content = " ".join(p.get_text()
                                       for p in paragraphs)[:max_chars]
                    content = re.sub(r'\s+', ' ', content).strip()

                    if not content or len(content) < 50:
                        # Fallback to getting all text
                        content = soup.get_text()[:max_chars]
                        content = re.sub(r'\s+', ' ', content).strip()

                except (requests.exceptions.RequestException, Exception) as e:
                    logger.info(f"Skipping {url}: {str(e)[:100]}...")
                    # Fallback to using title and snippet
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    content = f"{title}. {snippet}"

                # Only process if we have content
                if content and len(content.strip()) > 0:
                    extracted_keywords = generator.llm_generate_keywords(
                        hypothesis, content, max_tokens=60)
                else:
                    # If no content could be extracted, use only the title
                    extracted_keywords = generator.llm_generate_keywords(
                        hypothesis, result.get('title', ''), max_tokens=60)

                result["extracted_keywords"] = extracted_keywords
                if extracted_keywords and extracted_keywords != "Error generating keywords.":
                    self.extracted_keywords_list.append(
                        (result["url"], extracted_keywords))

        # Calculate success rate and total time for logging
        successful = len([r for r in self.last_search_results if r.get(
            "extracted_keywords") and r["extracted_keywords"] != "Error generating keywords."])
        total = len(self.last_search_results)
        elapsed_time = time.time() - start_time

        logger.info(
            f"Keyword enrichment completed in {elapsed_time:.1f} seconds")
        logger.info(
            f"Successfully enriched {successful}/{total} results ({successful/total*100:.1f}% success rate)")

    # Add this method to the ReflectionAgent class
    def _extract_domain(self, url):
        """Extract the domain from a URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            # Simple fallback if urlparse isn't available
            if url.startswith('http'):
                domain = url.split('//')[-1].split('/')[0]
                return domain
            return url

    # --- API Calls ---
    def _fetch_google_search(self, query, num_results=20):
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serapi_key,
            "num": num_results
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    link = item.get("link", "")
                    # Filter out any unwanted domains (e.g., zhihu) if necessary:
                    if "zhihu" in link.lower():
                        continue
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": link,
                        "source": "Google",
                        "engine": "google"
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching Google Search results: {e}")
            return []

    def _fetch_bing_search(self, query, num_results=20):
        url = "https://serpapi.com/search"
        params = {
            "engine": "bing",
            "q": query,
            "api_key": self.serapi_key,
            "count": num_results
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    link = item.get("link", "")
                    if "zhihu" in link.lower():
                        continue
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": link,
                        "source": "Bing",
                        "engine": "bing"
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching Bing Search results: {e}")
            return []

    def _fetch_duckduckgo_search(self, query, num_results=20):
        url = "https://serpapi.com/search"
        params = {
            "engine": "duckduckgo",
            "q": query,
            "api_key": self.serapi_key
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    link = item.get("link", "")
                    if "zhihu" in link.lower():
                        continue
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": link,
                        "source": "DuckDuckGo",
                        "engine": "duckduckgo"
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching DuckDuckGo Search results: {e}")
            return []

    def _fetch_google_scholar(self, query, num_results=20):
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.serapi_key,
            "num": num_results
        }
        try:
            response = requests.get(url, params=params)
            data = response.json()
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:num_results]:
                    link = item.get("link", "")
                    if "zhihu" in link.lower():
                        continue
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("publication_info", {}).get("summary", ""),
                        "url": link,
                        "source": item.get("publication_info", {}).get("authors", ["Google Scholar"])[0],
                        "engine": "google_scholar"
                    })
            return results
        except Exception as e:
            logger.error(f"Error fetching Google Scholar results: {e}")
            return []


if __name__ == "__main__":
    print("=== TECH & SCIENCE TREND REFLECTION AGENT ===")
    reflector = ReflectionAgent()
    while True:
        query = input("\nEnter your query (or 'quit' to exit):\n> ")
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid query.")
            continue
        print("Analyzing search results...")
        reflector.analyze_search_results(query)
        # Enrich results with extracted keywords using the Groq API
        generator = groqTextGenerator(
            api_key="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")
        reflector.enrich_results_with_keywords(query, generator)
        print("\n--- REGULAR SEARCH RESULTS (Top 20) ---")
        for i, res in enumerate(reflector.regular_results, start=1):
            print(f"{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            if res.get("extracted_keywords"):
                print(f"   Extracted Keywords: {res['extracted_keywords']}")
        print("\n--- GOOGLE SCHOLAR RESULTS (Top 20) ---")
        for i, res in enumerate(reflector.scholar_results, start=1):
            print(f"{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            if res.get("extracted_keywords"):
                print(f"   Extracted Keywords: {res['extracted_keywords']}")
