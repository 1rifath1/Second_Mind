import os
import json
import datetime
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple


class MemoryAgent:
    """
    Memory Agent implements both working and long-term memory for the Second Mind system
    using Pinecone for vector storage and similarity search.

    Features:
    - Stores interactions with embeddings for similarity search
    - Maintains working memory (recent interactions) and long-term memory
    - Provides retrieval mechanisms for the Proximity agent
    - Integrates with existing agent outputs
    """

    def __init__(self,
                 api_key: str = "pcsk_2yFRRb_GZQShHjGazfM6TTSdLGKDA4iUSaiRkB3yKt7q6mvUHDFwkZkLiJuYWQ6uCdM7Vk",
                 index_name: str = "second-mind",
                 model_name: str = "all-MiniLM-L6-v2",
                 working_memory_days: int = 7):
        """
        Initialize the Memory Agent with Pinecone vector database.

        Args:
            api_key: Pinecone API key
            index_name: Pinecone index name
            model_name: The name of the sentence transformer model to use
            working_memory_days: Number of days to keep items in working memory
        """
        # Load the sentence transformer model for embeddings
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Pinecone with new API
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)

        # Connect to existing index - no need to check/create since it already exists
        self.index = self.pc.Index(self.index_name)

        # Storage for memory entries that aren't in Pinecone
        self.memory_entries = {}
        self.working_memory_days = working_memory_days

        # Ensure storage directories exist
        os.makedirs("memory", exist_ok=True)

        # Load existing memory if available
        self._load_memory()

    def store_interaction(self,
                          query: str,
                          hypothesis: str,
                          coherence_results: Dict[str, Any],
                          ranked_keywords: List[Tuple[str, Dict[str, Any]]],
                          search_results: List[Dict[str, Any]],
                          agent_outputs: Dict[str, Any] = None) -> str:
        """
        Store a complete interaction in memory.

        Args:
            query: The original user query
            hypothesis: The generated hypothesis
            coherence_results: Results from the Coherence Checker
            ranked_keywords: Keywords ranked by the Ranking Agent
            search_results: Enriched search results from the Reflection Agent
            agent_outputs: Outputs from other agents (optional)

        Returns:
            str: The memory ID of the stored interaction
        """
        # Create a unique memory ID with timestamp
        timestamp = datetime.datetime.now().isoformat()
        memory_id = f"memory_{timestamp.replace(':', '_')}"

        # Create the text to embed (query + hypothesis + keywords)
        keywords = [kw for kw, _ in ranked_keywords[:5]
                    ] if ranked_keywords else []
        text_to_embed = f"{query} {hypothesis} {' '.join(keywords)}"

        # Generate the embedding
        embedding = self.model.encode([text_to_embed])[0].tolist()

        # Pad the embedding to 1024 dimensions if needed (since your index is 1024D)
        if len(embedding) < 1024:
            # Padding with zeros
            embedding = embedding + [0.0] * (1024 - len(embedding))
        elif len(embedding) > 1024:
            # Truncate to 1024 dimensions
            embedding = embedding[:1024]

        # Create memory entry
        memory_entry = {
            "id": memory_id,
            "timestamp": timestamp,
            "query": query,
            "hypothesis": hypothesis,
            "coherence_results": coherence_results,
            "ranked_keywords": [
                {
                    "keyword": kw,
                    "frequency": data["total_frequency"],
                    # Store top 5 references only
                    "references": data["references"][:5]
                } for kw, data in ranked_keywords
            ] if ranked_keywords else [],
            "search_results_sample": [
                {
                    "title": res["title"],
                    "url": res["url"],
                    "extracted_keywords": res.get("extracted_keywords", "")
                } for res in search_results[:10] if res.get("extracted_keywords")
            ] if search_results else [],
            "agent_outputs": agent_outputs,
            "memory_type": "working"  # Mark as working memory
        }

        # Store the full entry locally
        self.memory_entries[memory_id] = memory_entry

        # Store in Pinecone with metadata
        metadata = {
            "id": memory_id,
            "timestamp": timestamp,
            "query": query,
            "memory_type": "working"
        }

        # Add top keywords to metadata for filtering
        for i, (kw, _) in enumerate(ranked_keywords[:3]):
            if kw:
                metadata[f"keyword_{i}"] = kw

        self.index.upsert(
            vectors=[(memory_id, embedding, metadata)]
        )

        # Periodically manage memory (move old items to long-term)
        self._manage_memory()

        # Save memory entries
        self._save_memory()

        return memory_id

    def find_similar(self,
                     query: str,
                     k: int = 3,
                     include_long_term: bool = True) -> List[Dict[str, Any]]:
        """
        Find similar interactions based on the query.

        Args:
            query: The query to find similar interactions for
            k: Number of similar interactions to return
            include_long_term: Whether to include long-term memory in the search

        Returns:
            List of similar interactions with similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].tolist()

        # Pad the embedding to 1024 dimensions if needed
        if len(query_embedding) < 1024:
            # Padding with zeros
            query_embedding = query_embedding + \
                [0.0] * (1024 - len(query_embedding))
        elif len(query_embedding) > 1024:
            # Truncate to 1024 dimensions
            query_embedding = query_embedding[:1024]

        # Setup filter for memory type if not including long-term
        filter_dict = None
        if not include_long_term:
            filter_dict = {"memory_type": "working"}

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter_dict
        )

        similar_interactions = []
        for match in results.matches:
            memory_id = match.id
            similarity = match.score

            # Get the full memory entry
            if memory_id in self.memory_entries:
                entry = self.memory_entries[memory_id].copy()
                entry["similarity"] = float(similarity)
                similar_interactions.append(entry)

        # Sort by similarity (highest first)
        similar_interactions.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_interactions[:k]

    def get_all_keywords(self) -> Dict[str, int]:
        """
        Get all keywords from memory with their frequencies.

        Returns:
            Dict mapping keywords to their frequencies
        """
        keyword_frequencies = {}

        # Process all memory entries
        for memory_id, entry in self.memory_entries.items():
            for keyword_data in entry.get("ranked_keywords", []):
                keyword = keyword_data.get("keyword")
                if keyword:
                    keyword_frequencies[keyword] = keyword_frequencies.get(
                        keyword, 0) + 1

        return keyword_frequencies

    def _manage_memory(self):
        """
        Manage memory by moving old items from working to long-term memory.
        """
        current_time = datetime.datetime.now()
        cutoff_time = current_time - \
            datetime.timedelta(days=self.working_memory_days)

        ids_to_update = []

        # Find memory entries to update
        for memory_id, entry in self.memory_entries.items():
            if entry.get("memory_type") == "working":
                entry_time = datetime.datetime.fromisoformat(
                    entry["timestamp"])
                if entry_time < cutoff_time:
                    ids_to_update.append(memory_id)
                    # Update local entry
                    entry["memory_type"] = "long_term"

        # Batch update Pinecone
        if ids_to_update:
            # Get existing vectors to update metadata
            fetch_response = self.index.fetch(ids=ids_to_update)

            # Update vectors with new memory type
            update_vectors = []
            for memory_id in ids_to_update:
                if memory_id in fetch_response.vectors:
                    vector_data = fetch_response.vectors[memory_id]
                    # Create new metadata with updated memory type
                    metadata = vector_data.metadata.copy()
                    metadata["memory_type"] = "long_term"

                    # Add to update list
                    update_vectors.append(
                        (memory_id, vector_data.values, metadata)
                    )

            # Update in batches of 100 (Pinecone limit)
            batch_size = 100
            for i in range(0, len(update_vectors), batch_size):
                batch = update_vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)

    def _save_memory(self):
        """
        Save memory entries to disk.
        """
        with open("memory/memory_entries.json", "w") as f:
            json.dump(self.memory_entries, f, indent=2)

    def _load_memory(self):
        """
        Load memory entries from disk if available.
        """
        try:
            if os.path.exists("memory/memory_entries.json"):
                with open("memory/memory_entries.json", "r") as f:
                    self.memory_entries = json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory_entries = {}

    def export_memory(self, filename: str) -> bool:
        """
        Export complete memory to a single file.

        Args:
            filename: The filename to export to

        Returns:
            bool: Success status
        """
        try:
            # Create export data
            export_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "memory_entries": self.memory_entries
            }

            # Write to file
            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error exporting memory: {e}")
            return False

    def import_memory(self, filename: str) -> bool:
        """
        Import memory from a file.

        Args:
            filename: The filename to import from

        Returns:
            bool: Success status
        """
        try:
            # Read from file
            with open(filename, "r") as f:
                import_data = json.load(f)

            # Validate import data
            if "memory_entries" not in import_data:
                print("Error: Invalid memory export file format")
                return False

            # Set memory entries
            imported_entries = import_data["memory_entries"]

            # Update Pinecone with imported entries
            update_vectors = []
            for memory_id, entry in imported_entries.items():
                # Skip if already exists
                if memory_id in self.memory_entries:
                    continue

                # Get or create embedding
                if "embedding" in entry:
                    embedding = entry["embedding"]
                else:
                    # Generate embedding from text
                    keywords = [kw_data["keyword"]
                                for kw_data in entry.get("ranked_keywords", [])][:5]
                    text_to_embed = f"{entry['query']} {entry['hypothesis']} {' '.join(keywords)}"
                    embedding = self.model.encode([text_to_embed])[0].tolist()

                    # Pad to 1024 dimensions
                    if len(embedding) < 1024:
                        embedding = embedding + [0.0] * (1024 - len(embedding))
                    elif len(embedding) > 1024:
                        embedding = embedding[:1024]

                # Create metadata
                metadata = {
                    "id": memory_id,
                    "timestamp": entry["timestamp"],
                    "query": entry["query"],
                    "memory_type": entry.get("memory_type", "long_term")
                }

                # Add top keywords to metadata
                for i, kw_data in enumerate(entry.get("ranked_keywords", [])[:3]):
                    if "keyword" in kw_data:
                        metadata[f"keyword_{i}"] = kw_data["keyword"]

                # Add to update list
                update_vectors.append(
                    (memory_id, embedding, metadata)
                )

            # Update Pinecone in batches
            batch_size = 100
            for i in range(0, len(update_vectors), batch_size):
                batch = update_vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)

            # Update local entries
            self.memory_entries.update(imported_entries)

            # Save to disk
            self._save_memory()

            return True
        except Exception as e:
            print(f"Error importing memory: {e}")
            return False

    def clear_memory(self, confirm: bool = False) -> bool:
        """
        Clear all memory entries (but keep the index structure).

        Args:
            confirm: Confirmation flag (must be True to proceed)

        Returns:
            bool: Success status
        """
        if not confirm:
            print("Memory clear aborted. Set confirm=True to proceed.")
            return False

        try:
            # Delete all vectors from the index
            # Note: This keeps the index configuration but removes all data
            self.index.delete(delete_all=True)

            # Clear local entries
            self.memory_entries = {}

            # Save empty entries to disk
            self._save_memory()

            return True
        except Exception as e:
            print(f"Error clearing memory: {e}")
            return False
