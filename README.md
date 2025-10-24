# QueryMind

> Multi-agent RAG system with intelligent query routing, semantic search, and web fallback

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen.svg)](./tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

QueryMind is a lightweight, production-ready Retrieval-Augmented Generation (RAG) system that combines ChromaDB vector search, Ollama LLM intelligence, and web search capabilities to provide accurate, context-aware responses from your knowledge base.

## ✨ Features

- **🤖 Intelligent Query Routing** - Automatically routes queries to the optimal search strategy
- **🔍 Semantic Search** - ChromaDB-powered vector search with mxbai-embed-large embeddings
- **💡 LLM Intent Analysis** - Ollama integration for query understanding and keyword extraction
- **🌐 Web Search Fallback** - Seamless fallback to Serper.dev when vault has no results
- **📊 Structured Logging** - Environment-based logging with debug, info, warning, error levels
- **🛡️ Security Hardened** - Input sanitization, injection protection, and validation
- **🧪 Fully Tested** - 27 tests covering imports, routing logic, and security
- **📦 Pip Installable** - Standard Python package with pyproject.toml

## 🏗️ Architecture

QueryMind implements a **multi-agent architecture** with intelligent routing:

```
User Query → Router → [ Fast Search Agent   ] → Results
                      [ Deep Research Agent ]
                      [ Web Search (fallback) ]
```

### Agent Types

1. **FastSearchAgent** - Direct keyword matching for simple queries (<1s)
2. **DeepResearchAgent** - Ollama-powered semantic analysis for complex questions (~10s)
3. **WebSearchClient** - Serper.dev API integration for external knowledge

### Query Routing Heuristics

Queries are automatically routed based on:
- **Length**: >10 words → Deep Research
- **Question words**: "how", "why", "what", "explain" → Deep Research
- **Logical operators**: "and", "or", "not" → Deep Research
- **Default**: Simple keywords → Fast Search

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Ollama (running locally on port 11434)
- ChromaDB instance with indexed documents
- (Optional) Serper.dev API key for web search

### Installation

```bash
# Clone the repository
git clone https://github.com/rduffyuk/querymind.git
cd querymind

# Install the package
pip install .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Configure your environment variables:

```bash
# Required
VAULT_PATH=/path/to/your/obsidian-vault
CHROMADB_URL=http://localhost:8000

# Optional
SERPER_API_KEY=your-serper-api-key
OLLAMA_API_URL=http://localhost:11434
LOG_LEVEL=INFO
```

## 📖 Usage

### Basic Search

```python
from querymind import auto_search

# Simple query (uses FastSearchAgent)
result = auto_search("Redis caching")
print(f"Found {result.result_count} results")
for r in result.results:
    print(f"  - {r['file']}: {r['score']:.2f}")

# Complex query (uses DeepResearchAgent)
result = auto_search("How to implement Redis caching for APIs?")
print(f"Agent: {result.agent_type}")
print(f"Time: {result.elapsed_time:.2f}s")
```

### Advanced Usage

```python
from querymind.agents.router import AgentRouter

# Initialize router with custom configuration
router = AgentRouter(
    model="mistral:7b",
    enable_web_fallback=True
)

# Execute search with verbose logging
result = router.search(
    query="Explain StatefulSet vs Deployment",
    n_results=10,
    verbose=True
)

# Get routing statistics
stats = router.get_stats()
print(f"Fast searches: {stats['fast_searches']}")
print(f"Deep searches: {stats['deep_searches']}")
```

### Direct Agent Access

```python
from querymind.agents.vault_search_agent_local import VaultSearchAgentLocal
from querymind.agents.web_search_client import WebSearchClient

# Use vault search agent directly
vault_agent = VaultSearchAgentLocal(model="mistral:7b")
result = vault_agent.search("kubernetes deployment patterns")

# Use web search directly
web_client = WebSearchClient(api_key="your-key")
results = web_client.search_sync("latest FastAPI best practices", n_results=5)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_router_basic.py -v

# Run with coverage
pytest tests/ --cov=querymind --cov-report=html
```

Test coverage:
- ✅ 27 tests total
- ✅ 25 passing (92.6%)
- ⏭️ 2 skipped (optional dependencies)

### Test Suite

- **test_imports_work.py** - Verify all modules can be imported
- **test_router_basic.py** - Validate query routing logic and heuristics
- **test_security_validation.py** - Test input sanitization and injection protection

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/rduffyuk/querymind.git
cd querymind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Quality

```bash
# Format code
black querymind/ tests/

# Lint code
ruff querymind/ tests/
```

## 📋 Project Structure

```
querymind/
├── querymind/
│   ├── __init__.py           # Package initialization
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── logging_config.py # Structured logging
│   │   ├── embeddings.py     # ChromaDB embeddings
│   │   ├── cache.py          # Query caching (Redis)
│   │   ├── conversation_memory.py  # Conversation persistence (stub)
│   │   ├── feature_flags.py  # Feature flag management
│   │   └── search.py         # Search utilities
│   ├── agents/               # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Abstract base agent
│   │   ├── fast_search_agent.py    # Quick keyword search
│   │   ├── deep_research_agent.py  # LLM-powered search
│   │   ├── vault_search_agent_local.py  # Ollama integration
│   │   ├── web_search_client.py    # Web search fallback
│   │   └── router.py         # Intelligent routing
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── main.py
│   └── mcp/                  # Model Context Protocol
│       ├── server.py         # FastMCP server
│       └── security_validator.py  # Input validation
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_imports_work.py  # Import verification
│   ├── test_router_basic.py  # Routing logic tests
│   └── test_security_validation.py  # Security tests
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── LICENSE.txt               # MIT License
└── README.md                 # This file
```

## ⚙️ Configuration

QueryMind uses environment variables for configuration. See `.env.example` for all available options:

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `VAULT_PATH` | Path to your markdown documents | `/vault` |
| `CHROMADB_URL` | ChromaDB HTTP endpoint | `http://localhost:8000` |
| `REDIS_URL` | Redis cache endpoint | `redis://localhost:6379` |
| `OLLAMA_API_URL` | Ollama LLM endpoint | `http://localhost:11434` |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |

### Optional Features

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | [Serper.dev](https://serper.dev) API key for web search | None |
| `DISABLE_WEB_SEARCH` | Disable web fallback | `false` |

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add tests for new features
- Update documentation as needed
- Use structured logging (not print statements)

## 🗺️ Roadmap

### Current (v0.1.0)
- [x] Intelligent query routing with 7 heuristics
- [x] FastSearch, DeepResearch, WebSearch agents
- [x] Ollama integration for intent analysis
- [x] ChromaDB vector search
- [x] Structured logging system
- [x] Comprehensive test suite (27 tests)
- [x] Security hardening and input validation

### Planned (v0.2.0)
- [ ] Enhanced caching with gather cache
- [ ] Async support for concurrent searches
- [ ] Connection pooling for ChromaDB
- [ ] Advanced metrics and monitoring
- [ ] REST API endpoints
- [ ] Web UI for query testing

### Future (v1.0.0)
- [ ] Complete conversation memory implementation
- [ ] Hot-reload for configuration changes
- [ ] Docker Compose deployment
- [ ] Kubernetes deployment guides
- [ ] Multi-language support

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

QueryMind builds on excellent open-source projects:
- [ChromaDB](https://www.trychroma.com/) - Vector database for semantic search
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Serper.dev](https://serper.dev/) - Web search API
- [FastMCP](https://github.com/jlowin/fastmcp) - Model Context Protocol server

---

**QueryMind** - Intelligent search for your knowledge base

Made with ❤️ by [Ryan Duffy](https://github.com/rduffyuk)
