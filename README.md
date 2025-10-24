# QueryMind

<div align="center">

**Intelligence Behind Every Query**

🧠 Smart RAG system with intelligent query routing · 🚀 <1s for simple queries · 🔒 Local-first & private

[Quickstart](#-quickstart) • [Architecture](#-how-it-works) • [Examples](#-examples) • [Support](#-support)

</div>

---

## 🎯 What is QueryMind?

QueryMind is an **intelligent RAG (Retrieval-Augmented Generation) system** that automatically chooses the best search strategy for your query.

Not all questions are equal. QueryMind uses a **7-heuristic router** to decide:
- 🏃 **FastSearch** (<1s) - Simple keyword lookups via BM25 + vector search
- 🧠 **DeepResearch** (~10s) - Complex questions requiring LLM analysis
- 🌐 **WebSearch** (2-5s) - External knowledge via Google (Serper.dev API)

### Real-World Example

```python
from querymind import search

# Simple query → FastSearch (0.8s)
search("Redis caching patterns")

# Complex query → DeepResearch (12s, but comprehensive)
search("How should I architect Redis caching for a microservices system?")

# Unknown topic → WebSearch (3s)
search("Latest Redis features in 2025")
```

QueryMind **automatically routes** each query to the optimal agent. You just ask—it figures out the rest.

---

## 🚀 How to Use QueryMind

Choose your workflow:

### 🤖 **With Claude Code** (Recommended)
Add QueryMind as an MCP server - Claude Code automatically uses it when you ask questions about your vault.

```json
// ~/.claude/config.json
{
  "mcpServers": {
    "querymind": {
      "command": "docker",
      "args": ["exec", "querymind-mcp", "fastmcp", "run", "querymind.mcp.server"]
    }
  }
}
```

**Then just chat**:
- "Search my vault for Redis caching patterns"
- "Find recent project updates from this week"
- "What did I write about machine learning?"

### 💻 **From Terminal** (Local CLI)
Run QueryMind commands directly:

```bash
# Search your vault
python -m querymind.cli search "Redis caching"

# Ask complex questions
python -m querymind.cli ask "How to implement caching?"

# Index new documents
python -m querymind.cli index ~/Documents/vault
```

### 🐍 **Python Scripts** (Custom integrations)
```python
from querymind import search

results = search("Redis caching patterns", n_results=5)
for r in results['results']:
    print(f"{r['file']}: {r['score']}")
```

**📖 See [USER-GUIDE.md](USER-GUIDE.md) for detailed usage examples and workflows**

---

## ✨ Key Features

### 🎯 Intelligent Query Routing
- **7 heuristics** analyze query complexity in <50μs
- Automatic agent selection (fast/deep/web)
- 70% queries resolve in <1s (FastSearch)
- 25% use deep LLM analysis when needed
- 5% fall back to web for external knowledge

### ⚡ Performance
- **<1 second** for 70% of queries (FastSearch)
- **73% cache hit rate** (Redis with smart TTLs)
- **99.25% token reduction** via progressive disclosure
- **GPU-accelerated** embeddings (ChromaDB + mxbai-embed-large)

### 🔒 Privacy First (Local-First Design)
- **Core features 100% local** - FastSearch & DeepResearch run entirely on your machine
- **WebSearch optional** - Only used for external knowledge (e.g., "latest news 2025"), can be disabled
- **Zero telemetry** - Your vault data never leaves your machine
- **Transparent** - Full bash command logs, no hidden operations
- **Air-gapped compatible** - Works completely offline (disable WebSearch)

### 🛠️ Developer-Friendly
- **MCP (Model Context Protocol)** - 15 optimized tools for Claude Code/Aider
- **Docker Compose** - One command to start all services
- **RESTful API** - Easy integration with any application
- **5 examples** - From basic search to batch indexing

---

## 🚀 Quickstart

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU (optional, for faster embeddings)

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/querymind/querymind.git
cd querymind

# 2. Configure environment
cp docker/.env.example docker/.env
# Edit docker/.env with your preferences (optional)

# 3. Start all services
cd docker
docker-compose up -d

# 4. Wait for services to initialize (~2 minutes)
./scripts/health-check.sh

# 5. Index your first documents
python examples/01_basic_search.py
```

**That's it!** QueryMind is now running. Try the examples:

```bash
# Simple search
python examples/01_basic_search.py

# See intelligent routing in action
python examples/02_intelligent_routing.py

# Search with date filters
python examples/03_temporal_search.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

---

## ⚙️ Configuration

QueryMind uses environment variables for all configuration. Copy `.env.example` to `.env` and customize:

```bash
# Copy the example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `VAULT_PATH` | Path to your markdown documents | `/vault` |
| `CHROMADB_COLLECTION` | Vector database collection name | `obsidian_vault_mxbai` |

### Service URLs

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMADB_URL` | ChromaDB HTTP endpoint | `http://localhost:8000` |
| `REDIS_URL` | Redis cache endpoint | `redis://localhost:6379` |
| `OLLAMA_URL` | Ollama LLM endpoint | `http://localhost:11434` |

### Performance Tuning

| Variable | Description | Default |
|----------|-------------|---------|
| `ROUTER_FAST_THRESHOLD` | Word count to trigger DeepResearch | `10` |
| `CACHE_TTL_QUERY` | Query cache TTL (seconds) | `3600` (1 hour) |
| `CACHE_TTL_GATHER` | LLM analysis cache TTL (seconds) | `300` (5 min) |

### External APIs (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | [Serper.dev](https://serper.dev) API key for web search | None |
| `DISABLE_WEB_SEARCH` | Disable web fallback | `false` |

### Example Configurations

**Docker deployment:**
```bash
VAULT_PATH=/vault
CHROMADB_URL=http://chromadb:8000
REDIS_URL=redis://redis:6379
OLLAMA_URL=http://ollama:11434
```

**Local development:**
```bash
VAULT_PATH=/home/user/Documents/vault
CHROMADB_URL=http://localhost:8000
LOG_LEVEL=DEBUG
```

**Production:**
```bash
VAULT_PATH=/mnt/vault
SERPER_API_KEY=your_production_key
CACHE_TTL_QUERY=7200
LOG_LEVEL=INFO
```

---

## 🏗️ How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   User Query                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Router (7 heuristics)│  <50μs decision time
        └──────────┬─────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌──────────────────┐
│ FastSearch    │    │ DeepResearch     │
│ <1s (70%)     │    │ ~10s (25%)       │
│               │    │                  │
│ • BM25        │    │ • LLM analysis   │
│ • Vector      │    │ • Multi-doc      │
│ • Cache       │    │ • Synthesis      │
└───────────────┘    └──────────────────┘
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   ChromaDB (38K docs)│
        │   Redis Cache (73%)  │
        │   Ollama LLMs        │
        └─────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Routing** | Python (7 heuristics) | Intelligent query classification |
| **Search** | ChromaDB + BM25 | Hybrid semantic + keyword search |
| **LLM** | Ollama (mistral:7b, qwen2.5-coder:14b) | Local inference, zero cloud costs |
| **Cache** | Redis | 73% hit rate, 5min-1hr TTLs |
| **MCP** | FastMCP | 15 tools for Claude Code/Aider integration |
| **Storage** | ChromaDB | 38,380 documents, mxbai-embed-large embeddings |

See [ARCHITECTURE.md](docs/architecture.md) for technical deep-dive.

---

## 📚 Examples

### 1. Basic Search
```python
from querymind import search

results = search("machine learning pipelines", n_results=5)
for result in results['results']:
    print(f"{result['file']}: {result['score']}")
```

### 2. Intelligent Routing
```python
from querymind import auto_search

# Router automatically selects best agent
response = auto_search("How do I implement caching?", verbose=True)
print(f"Agent used: {response['agent_type']}")  # "deep_research"
print(f"Time: {response['elapsed_time']:.2f}s")  # ~12s
```

### 3. Temporal Search (Date Filtering)
```python
from querymind import search

# Search only documents from October 2025
results = search(
    "project updates",
    date_filter="2025-10-01",
    start_date="2025-10-01",
    end_date="2025-10-31"
)
```

See [examples/](examples/) for 5 complete examples with detailed comments.

---

## 🎨 MCP Integration (Claude Code / Aider)

QueryMind provides **15 optimized MCP tools** for seamless integration with Claude Code and Aider:

```json
{
  "mcpServers": {
    "querymind": {
      "command": "docker",
      "args": ["exec", "querymind-mcp", "fastmcp", "run", "querymind.mcp.server"]
    }
  }
}
```

**Available Tools**:
- `auto_search_vault` - Intelligent search with automatic routing
- `web_search_vault` - External knowledge via Google
- `explore_vault_structure` - Filesystem exploration
- `grep_vault_content` - Fast grep-based search
- `index_file_to_chromadb` - Real-time indexing
- `get_gpu_status` - Hardware monitoring
- ...and 9 more tools

See [docs/mcp-integration.md](docs/mcp-integration.md) for setup guide.

---

## ☕ Support

QueryMind is **free and open-source forever** (MIT License).

- ✅ **All features unlocked** - No paywalls, no restrictions
- ✅ **Personal & commercial use** - Use it however you want
- ✅ **No telemetry** - 100% private, no tracking
- ✅ **No account required** - Just download and run

**If QueryMind helps you**, consider supporting development:

**☕ [Buy Me a Coffee](https://buymeacoffee.com/rduffy)**

Your support helps with continued development, bug fixes, and infrastructure costs.

**Coming soon**: Optional commercial support plans for enterprises (inspired by [Obsidian's licensing](https://help.obsidian.md/teams/license)). For now, it's all free!

See [SUPPORT.md](SUPPORT.md) for more ways to help.

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

### Development Setup

```bash
# Clone repository
git clone https://github.com/querymind/querymind.git
cd querymind

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Start services in dev mode
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
```

---

## 📊 Performance Benchmarks

| Metric | Value | Context |
|--------|-------|---------|
| **FastSearch latency** | <1s | 70% of queries |
| **DeepResearch latency** | ~10s | 25% of queries (first time) |
| **DeepResearch (cached)** | <0.1s | 73% cache hit rate |
| **Router overhead** | <50μs | Query classification |
| **Documents indexed** | 38,380 | Default test corpus |
| **Embedding model** | mxbai-embed-large | 1024 dimensions |
| **Cache hit rate** | 73% | Redis with smart TTLs |

Run your own benchmarks:
```bash
./scripts/benchmark.sh
```

---

## 🗺️ Roadmap

### v1.0 (Current)
- [x] Intelligent query routing (7 heuristics)
- [x] FastSearch, DeepResearch, WebSearch agents
- [x] MCP server with 15 tools
- [x] Docker Compose deployment
- [x] Redis caching (73% hit rate)

### v1.1 (Next 30 days)
- [ ] REST API with OpenAPI spec
- [ ] Web UI for query testing
- [ ] PostgreSQL support (alternative to ChromaDB)
- [ ] Multi-language support (Python, JavaScript, Go clients)

### v2.0 (3-6 months)
- [ ] Custom agent plugins
- [ ] Distributed deployment (Kubernetes)
- [ ] Team collaboration features
- [ ] Advanced analytics dashboard

See [GitHub Issues](https://github.com/querymind/querymind/issues) for detailed roadmap.

---

## 📄 License

QueryMind is released under the **MIT License** - free for personal and commercial use.

See [LICENSE.txt](LICENSE.txt) for full details.

**Support the project**: ☕ [Buy Me a Coffee](https://buymeacoffee.com/rduffy)

---

## 🙏 Acknowledgments

QueryMind builds on excellent open-source projects:
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Ollama](https://github.com/ollama/ollama) - Local LLM inference
- [FastMCP](https://github.com/jlowin/fastmcp) - Model Context Protocol server
- [Anthropic](https://www.anthropic.com/) - Claude Code integration inspiration

---

## 📞 Get Help

- 📖 **Documentation**: [docs/](docs/)
- 💬 **Discord**: [Join our community](https://discord.gg/querymind) (coming soon)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/querymind/querymind/issues)
- 📧 **Email**: hello@querymind.dev

---

<div align="center">

**Made with ❤️ for the AI community**

[⭐ Star us on GitHub](https://github.com/querymind/querymind) · [☕ Buy Me a Coffee](https://buymeacoffee.com/rduffy) · [💼 LinkedIn](https://www.linkedin.com/in/rduffyuk) · [📝 Blog](https://blog.rduffy.uk) · [🐦 Twitter](https://twitter.com/querymind)

</div>
