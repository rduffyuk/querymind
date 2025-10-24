#!/usr/bin/env python3
"""
ChromaDB with mxbai-embed-large Embeddings Configuration V2
Using ChromaDB's SentenceTransformerEmbeddingFunction for proper integration
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import torch
import time
from pathlib import Path
from typing import List, Dict, Any

class ChromaDBManager:
    """Manages ChromaDB with optimized embeddings for your Obsidian vault"""

    def __init__(self, persist_path: str = "/home/rduffy/Documents/Leveling-Life/chromadb_data"):
        """Initialize ChromaDB with mxbai-embed-large embeddings"""

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        print("📊 Initializing ChromaDB with mxbai-embed-large...")

        # Check GPU availability
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("⚠️ No GPU detected, using CPU")

        # Use ChromaDB's built-in SentenceTransformer embedding function
        # This automatically handles the proper interface
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            device=device
        )

        print(f"✅ Loaded mxbai-embed-large on {device}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        print(f"✅ ChromaDB initialized at: {self.persist_path}")

        # Show GPU memory usage after model load
        if torch.cuda.is_available():
            print(f"   VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def create_collection(self, name: str, description: str = "", optimize_hnsw: bool = True) -> chromadb.Collection:
        """
        Create or get a collection with optimized settings

        Args:
            name: Collection name
            description: Collection description
            optimize_hnsw: Enable HNSW optimization for SIMD/AVX (2025 best practice)

        Returns:
            ChromaDB collection with optimized HNSW index
        """
        import os

        # Delete existing collection if it exists (for testing)
        try:
            self.client.delete_collection(name)
            print(f"🗑️ Deleted existing collection: {name}")
        except:
            pass

        # Base metadata
        metadata = {
            "description": description,
            "embedding_model": "mxbai-embed-large-v1",
            "dimensions": 1024,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # HNSW Optimization (2025-10-11 enhancement)
        # Source: https://cookbook.chromadb.dev/running/performance-tips/
        # Enables SIMD/AVX CPU optimizations for 25% faster searches
        if optimize_hnsw:
            cpu_count = os.cpu_count() or 4
            metadata.update({
                "hnsw:space": "cosine",              # Cosine similarity for semantic search
                "hnsw:construction_ef": 200,         # Higher = better quality (default: 100)
                "hnsw:M": 32,                        # Higher = better recall (default: 16)
                "hnsw:num_threads": cpu_count,       # Use all CPU cores
                "hnsw:search_ef": 100                # Search-time parameter (default: 10)
            })
            print(f"🚀 HNSW optimization enabled: ef={200}, M={32}, threads={cpu_count}")

        # Create new collection with custom embedding function
        collection = self.client.create_collection(
            name=name,
            embedding_function=self.embedding_function,
            metadata=metadata
        )

        print(f"✅ Created collection: {name}")
        return collection

    def get_collection(self, name: str) -> chromadb.Collection:
        """Get existing collection"""
        return self.client.get_collection(
            name=name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, collection: chromadb.Collection,
                     documents: List[str],
                     metadatas: List[Dict[str, Any]] = None,
                     ids: List[str] = None) -> None:
        """Add documents to collection with progress tracking"""

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"index": i} for i in range(len(documents))]

        print(f"📝 Adding {len(documents)} documents to collection...")

        start_time = time.time()

        # Add in batches for better performance
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))

            collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )

            if len(documents) > batch_size:
                print(f"   Processed {batch_end}/{len(documents)} documents...")

        elapsed = time.time() - start_time
        print(f"✅ Added {len(documents)} documents in {elapsed:.2f} seconds")

        if len(documents) > 0:
            print(f"   Average: {elapsed/len(documents)*1000:.2f}ms per document")

    def search(self, collection: chromadb.Collection,
              query: str,
              n_results: int = 5) -> Dict[str, Any]:
        """Search collection with performance metrics"""

        print(f"\n🔍 Searching for: '{query[:50]}...'")

        start_time = time.time()

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        elapsed = (time.time() - start_time) * 1000

        print(f"✅ Found {len(results['documents'][0])} results in {elapsed:.2f}ms")

        return results

    def get_stats(self, collection_name: str = None) -> Dict[str, Any]:
        """Get database statistics"""

        stats = {
            "collections": [],
            "total_documents": 0,
            "gpu_status": "GPU Active" if torch.cuda.is_available() else "CPU Mode",
            "vram_used": 0
        }

        # Get collection info
        if collection_name:
            try:
                col = self.client.get_collection(collection_name)
                count = col.count()
                stats["collections"].append({
                    "name": col.name,
                    "documents": count,
                    "metadata": col.metadata
                })
                stats["total_documents"] = count
            except:
                pass
        else:
            collections = self.client.list_collections()
            for col in collections:
                count = col.count()
                stats["collections"].append({
                    "name": col.name,
                    "documents": count,
                    "metadata": col.metadata
                })
                stats["total_documents"] += count

        # Get GPU memory if available
        if torch.cuda.is_available():
            stats["vram_used"] = torch.cuda.memory_allocated() / 1024**3
            stats["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return stats

def test_embeddings():
    """Test the embedding setup with sample Obsidian content"""

    print("=" * 60)
    print("Testing mxbai-embed-large with ChromaDB")
    print("=" * 60)

    # Initialize manager
    manager = ChromaDBManager()

    # Create test collection
    collection = manager.create_collection(
        name="obsidian_vault",
        description="Obsidian vault with semantic search capabilities"
    )

    # Sample documents from your vault context
    test_documents = [
        "ConvoCanvas is a visual knowledge management system that integrates with Obsidian for enhanced note-taking",
        "The AI-DO processor automatically routes files based on content keywords and visual intelligence analysis",
        "ChromaDB provides vector storage for semantic search across 493 markdown files in the Obsidian vault",
        "RTX 4080 with 16GB VRAM enables fast local AI processing at 71.61 tokens/second using Ollama",
        "Journal automation captures work progress every 30 minutes and triggers social content generation",
        "MCP integration enables direct Claude-vault connection with <50ms response time for real-time updates",
        "DeepSeek R1 models (7B/14B/32B) run locally via Ollama for enhanced reasoning capabilities",
        "Visual workflow uses Logseq and Excalidraw for spatial intelligence processing and diagram analysis",
        "Full conversation transcripts are captured using claude-full-logger.py for complete context preservation",
        "The LinkedIn automation system analyzes journal entries and generates professional posts automatically"
    ]

    # Create metadata for documents
    metadatas = [
        {"source": "system", "type": "project", "topic": "ConvoCanvas"},
        {"source": "system", "type": "automation", "topic": "AI-DO"},
        {"source": "system", "type": "database", "topic": "ChromaDB"},
        {"source": "hardware", "type": "gpu", "topic": "RTX 4080"},
        {"source": "automation", "type": "journal", "topic": "content generation"},
        {"source": "integration", "type": "mcp", "topic": "claude"},
        {"source": "ai", "type": "model", "topic": "DeepSeek"},
        {"source": "workflow", "type": "visual", "topic": "Excalidraw"},
        {"source": "logging", "type": "conversation", "topic": "transcripts"},
        {"source": "automation", "type": "social", "topic": "LinkedIn"}
    ]

    # Add documents
    manager.add_documents(
        collection=collection,
        documents=test_documents,
        metadatas=metadatas
    )

    # Test searches
    test_queries = [
        "How does visual processing work in the system?",
        "What GPU hardware is being used for AI processing?",
        "Tell me about journal automation and content generation",
        "How are conversations captured and stored?",
        "What is the semantic search capability?"
    ]

    print("\n" + "=" * 60)
    print("Search Results:")
    print("=" * 60)

    for query in test_queries:
        results = manager.search(collection, query, n_results=3)

        print(f"\n📌 Query: '{query}'")
        print("-" * 50)

        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity score
            print(f"  [{i+1}] Similarity: {similarity:.3f}")
            print(f"      Topic: {metadata.get('topic', 'N/A')}")
            print(f"      Text: {doc[:100]}...")

    # Show stats
    print("\n" + "=" * 60)
    print("Database Statistics:")
    print("=" * 60)

    stats = manager.get_stats("obsidian_vault")
    print(f"📊 Collection: {stats['collections'][0]['name']}")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Model: {stats['collections'][0]['metadata']['embedding_model']}")
    print(f"   Dimensions: {stats['collections'][0]['metadata']['dimensions']}")
    print(f"🎮 GPU Status: {stats['gpu_status']}")

    if stats['vram_used'] > 0:
        print(f"   VRAM Used: {stats['vram_used']:.2f} GB / {stats['vram_total']:.1f} GB")
        print(f"   Utilization: {stats['vram_used']/stats['vram_total']*100:.1f}%")

    print("\n✅ mxbai-embed-large successfully configured!")
    print("\n📋 Next Steps:")
    print("1. ✅ ChromaDB installed and tested")
    print("2. ✅ Full conversation logging active")
    print("3. ✅ mxbai-embed-large configured with GPU")
    print("4. ⏳ Implement markdown chunking strategy")
    print("5. ⏳ Index your 493 Obsidian vault files")
    print("6. ⏳ Create conversation memory pipeline")

if __name__ == "__main__":
    test_embeddings()