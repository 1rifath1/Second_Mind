# The Second Mind

A system of specialized AI agents that work together to generate, refine, and evolve scientific hypotheses by incorporating real-time web data and maintaining memory of past interactions.

## 🧠 Project Overview

The Second Mind is a coalition of specialized agents that mimic human learning - retaining preferences, connecting ideas, and improving with each interaction. The system extracts information from the web in real-time to inform research and shows iterative improvement in scientific reasoning and hypothesis generation.

## 🤖 Agent Architecture

The system consists of seven specialized agents, each with a distinct role, orchestrated by a Supervisor:

### Core Agents

1. **Generation Agent**: Creates initial hypotheses based on user queries.
   - Uses the Groq LLM API for text generation
   - Includes a coherence checker to evaluate the quality of generated hypotheses

2. **Reflection Agent**: Fetches and enriches search results from the web.
   - Fetches results from Google, Bing, DuckDuckGo, and Google Scholar
   - Enriches results with extracted keywords related to the hypothesis

3. **Ranking Agent**: Scores keywords and outputs based on relevance.
   - Implements a forced cycling mechanism to ensure diversity in keyword selection
   - Provides comprehensive details for top keywords

4. **Evolution Agent**: Refines ideas based on all available data.
   - Uses the Groq API to generate refined hypotheses
   - Incorporates web data, ranked keywords, and proximity data
   - Can further refine hypotheses with specific focus areas

5. **Proximity Agent**: Finds similar past interactions.
   - Works with the Memory Agent to retrieve relevant past queries
   - Extracts useful information from similar interactions
   - Provides adaptation suggestions based on past successes

6. **Memory Agent**: Stores and retrieves past interactions.
   - Uses Pinecone vector database for similarity search
   - Maintains both working memory (recent interactions) and long-term memory
   - Provides retrieval mechanisms based on embeddings

7. **Meta-Review Agent**: Evaluates the entire system's process.
   - Identifies bottlenecks and tracks performance metrics
   - Suggests improvements for future cycles
   - Maintains historical metrics for ongoing optimization

### Supervisor

The **Supervisor Agent** orchestrates all specialized agents:
   - Manages the flow between agents and allocates resources dynamically
   - Enables feedback loops for continuous improvement
   - Provides an interactive interface for users

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (install via pip):
  - requests
  - beautifulsoup4
  - pinecone-client
  - numpy
  - sentence-transformers
  - logging

### API Keys

You'll need the following API keys to run the system:
- Groq API key for text generation
- SerpAPI key for web searches
- Pinecone API key for vector database

Either set them as environment variables or update them directly in the code:
```
export GROQ_API_KEY="your_groq_api_key"
export SERAPI_KEY="your_serapi_key"
export PINECONE_API_KEY="your_pinecone_api_key"
```

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/second-mind.git
cd second-mind
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### Running the System

Run the Supervisor Agent to start the interactive system:

```
python supervisor_agent.py
```

## 🔄 System Workflow

1. **User enters a query** (e.g., "Renewable energy for urban areas")
2. **Supervisor** orchestrates the following process:
   - **Proximity Agent** checks for similar past queries
   - **Generation Agent** creates an initial hypothesis
   - **Reflection Agent** fetches and enriches search results from the web
   - **Ranking Agent** scores keywords based on search result relevance
   - **Evolution Agent** refines the hypothesis based on all gathered data
   - **Memory Agent** stores the interaction for future reference
   - **Meta-Review Agent** evaluates the process and suggests improvements
3. **Results are displayed** to the user, including the initial and refined hypotheses

## 💾 Data Storage

The system maintains several types of storage:
- **In-memory storage** for the current session
- **File-based storage** for saving results and memory entries
- **Pinecone vector database** for similarity search and retrieval

Results are saved to the `results/` directory with the cycle ID as a subfolder.
Memory entries are saved to the `memory/` directory.

## 🔍 Interactive Commands

When running the system interactively, you can use the following commands:
- `/memory on` - Enable memory features
- `/memory off` - Disable memory features
- `/detail on` - Enable detailed output
- `/detail off` - Disable detailed output
- `/stats` - Show system statistics
- `/help` - Show help message
- `exit`, `quit`, `q` - Exit the system

## 📝 Example Usage

```
=== THE SECOND MIND - INTERACTIVE MODE ===
Welcome to The Second Mind system! Enter your queries below.
Type 'exit' or 'quit' to exit the system.

Enter your query: Potential health benefits of intermittent fasting

Processing query: 'Potential health benefits of intermittent fasting'
This may take a moment as multiple agents are working together...

=== INITIAL HYPOTHESIS ===
Intermittent fasting may improve metabolic health by promoting cellular repair mechanisms, enhancing insulin sensitivity, and reducing inflammation, potentially leading to weight loss, improved cardiovascular health, and extended lifespan in humans.
Coherence: 8.3/10 - Excellent
Keywords: intermittent fasting, metabolic health, cellular repair, insulin sensitivity, inflammation

=== RANKED KEYWORDS ===
1. intermittent fasting (frequency: 85)
2. insulin sensitivity (frequency: 42)
3. metabolic health (frequency: 38)
4. inflammation (frequency: 29)
5. cellular repair (frequency: 15)

=== REFINED HYPOTHESIS ===
Intermittent fasting may improve metabolic health by activating autophagy (cellular repair), enhancing insulin sensitivity, and reducing chronic inflammation, with recent studies showing potential benefits for cardiovascular disease risk reduction, cognitive function, and longevity when implemented in structured patterns like 16:8 or 5:2 regimens.

Explanation of improvements:
The refined hypothesis incorporates recent scientific evidence on specific intermittent fasting protocols (16:8 and 5:2) and expands the potential benefits to include cognitive function. The cellular repair mechanism is more precisely identified as autophagy, and the hypothesis now specifies that inflammation reduction is specifically related to chronic inflammation. The cardiovascular benefit is more specifically defined as risk reduction.

Specific improvements made:
1. Specified the cellular repair mechanism as "autophagy"
2. Added specific fasting protocols (16:8 and 5:2) that are supported by research
3. Expanded potential benefits to include cognitive function based on recent studies

Cycle ID: 20240327_123456
```

## 📂 Project Structure

```
second-mind/
├── generation_agent.py    # Initial hypothesis generation
├── reflection_agent.py    # Web data extraction and analysis
├── ranking_agent.py       # Keyword ranking and scoring
├── evolution_agent.py     # Hypothesis refinement
├── proximity_agent.py     # Similar interaction retrieval
├── memory_agent.py        # Storage and retrieval mechanisms
├── metareview_agent.py    # System evaluation and optimization
├── supervisor_agent.py    # Orchestration and user interface
├── results/               # Saved results from each cycle
└── memory/                # Memory storage files
```

## 🔒 API Keys and Security

The project uses several external APIs:
- Groq API for text generation
- SerpAPI for web search
- Pinecone for vector storage

API keys are included in the code for demonstration purposes but should be replaced with your own keys or managed through environment variables for security in production.

## 🧪 Future Improvements

- Implement more advanced caching mechanisms for search results
- Add support for more search engines and knowledge sources
- Improve the memory management with forgetting mechanisms
- Add visualization tools for the agent interactions
- Develop more specialized agents for specific domains
- Implement a web interface for easier interaction

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Groq for providing the LLM API
- SerpAPI for search capabilities
- Pinecone for vector database services

---

Created as part of the AI Learning System Challenge
