#!/usr/bin/env python3
"""
QueryMind Example 1: Basic Search

Demonstrates simple semantic search using QueryMind.

Before running:
1. Start services: cd docker && docker-compose up -d
2. Wait for services to be healthy
3. Ensure your vault is indexed in ChromaDB

Usage:
    python examples/01_basic_search.py
"""

from querymind import search

def main():
    print("🔍 QueryMind - Basic Search Example\n")
    print("=" * 60)

    # Example query
    query = "Redis caching patterns"
    print(f"\nSearching for: '{query}'")
    print("-" * 60)

    try:
        # Execute search
        result = search(query, n_results=5)

        if result.status == "success" and result.results:
            print(f"\n✅ Found {result.result_count} results in {result.elapsed_time:.2f}s")
            print(f"🤖 Agent used: {result.agent_type}\n")

            # Display results
            for i, doc in enumerate(result.results, 1):
                print(f"{i}. {doc['file']}")
                print(f"   Score: {doc['score']:.3f}")
                print(f"   Preview: {doc['content'][:100]}...")
                if doc.get('cached'):
                    print(f"   💾 [Cached]")
                print()

        else:
            print(f"\n⚠️  No results found")
            if result.error:
                print(f"Error: {result.error}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Docker services are running (docker-compose up -d)")
        print("  2. ChromaDB has indexed documents")
        print("  3. Redis is accessible")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
