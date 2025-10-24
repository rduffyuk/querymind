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

**System Requirements:**
- Python 3.9 or higher
- 8GB+ RAM (16GB recommended for better performance)
- (Optional) NVIDIA GPU for faster embeddings

**Required Services:**
- **Ollama** - Local LLM inference (mistral:7b or similar)
- **ChromaDB** - Vector database for semantic search
- **Redis** - Query caching (optional but recommended)

### Step 1: Install Ollama

Ollama provides local LLM inference for query analysis.

**macOS / Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the mistral model (7B parameters, ~4GB)
ollama pull mistral:7b

# Verify installation
ollama list
```

**Windows:**
1. Download installer from https://ollama.com/download
2. Run installer and follow prompts
3. Open PowerShell and run: `ollama pull mistral:7b`

**Verify Ollama is running:**
```bash
curl http://localhost:11434/api/tags
# Should return list of installed models
```

### Step 2: Install ChromaDB

ChromaDB provides vector search capabilities.

**Option A: Install as Python package (Recommended for development)**
```bash
# ChromaDB will be installed automatically with QueryMind
# It runs in-process (no separate server needed)
```

**Option B: Run ChromaDB server (Recommended for production)**
```bash
# Install ChromaDB server
pip install chromadb

# Run ChromaDB server
chroma run --host localhost --port 8000

# Verify server is running
curl http://localhost:8000/api/v1/heartbeat
```

### Step 3: Install Redis (Optional)

Redis provides query caching for better performance (73% cache hit rate).

**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

**Windows:**
```bash
# Download from https://github.com/microsoftarchive/redis/releases
# Or use WSL2 with Ubuntu instructions above
```

**Verify Redis:**
```bash
redis-cli ping
# Should return: PONG
```

### Step 4: Install QueryMind

```bash
# Clone the repository
git clone https://github.com/rduffyuk/querymind.git
cd querymind

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install QueryMind with all dependencies
pip install .

# Or install in development mode
pip install -e ".[dev]"
```

### Step 5: Configure Environment

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Required - Path to your markdown documents
VAULT_PATH=/path/to/your/obsidian-vault

# ChromaDB settings
CHROMADB_URL=http://localhost:8000  # Or leave blank for in-process mode

# Redis settings (optional - will fall back to in-memory cache)
REDIS_URL=redis://localhost:6379

# Ollama settings
OLLAMA_API_URL=http://localhost:11434

# Optional - Web search API key (100 free queries/month)
SERPER_API_KEY=your-api-key-here

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Step 6: Verify Installation

Run the test suite to verify everything is working:

```bash
# Run all tests
pytest tests/ -v

# Should see: 25 passed, 2 skipped
```

Test a simple query:

```python
from querymind import auto_search

# Simple test query
result = auto_search("test query", n_results=1)
print(f"Status: {result.status}")
print(f"Agent: {result.agent_type}")
```

### Optional: Get Serper.dev API Key

For web search fallback functionality:

1. Sign up at https://serper.dev
2. Get your API key from the dashboard
3. Add to `.env`: `SERPER_API_KEY=your-key-here`
4. Free tier: 100 queries/month
5. After free tier: $0.30 per 1,000 queries

### Optional: Install Obsidian for Document Management

Obsidian is a powerful markdown editor that works well for managing the document vault that QueryMind searches. While not required, it provides a great interface for creating and organizing your knowledge base.

**macOS:**
```bash
# Download from website
open https://obsidian.md/download

# Or install via Homebrew
brew install --cask obsidian
```

**Linux:**
```bash
# Download AppImage from website
wget https://github.com/obsidianmd/obsidian-releases/releases/download/v1.4.16/Obsidian-1.4.16.AppImage

# Make executable and run
chmod +x Obsidian-1.4.16.AppImage
./Obsidian-1.4.16.AppImage

# Or install via Snap
sudo snap install obsidian --classic
```

**Windows:**
```bash
# Download installer from website
start https://obsidian.md/download

# Or install via Chocolatey
choco install obsidian
```

**Setup your vault:**
1. Open Obsidian
2. Create a new vault or open existing vault at `VAULT_PATH` from your `.env`
3. Start creating markdown documents
4. QueryMind will automatically index and search these documents

### Troubleshooting

**Ollama connection failed:**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
# macOS/Linux: sudo systemctl restart ollama
# Windows: Restart Ollama Desktop app
```

**ChromaDB errors:**
```bash
# If using server mode, check if running
curl http://localhost:8000/api/v1/heartbeat

# If in-process mode, ensure adequate RAM
# ChromaDB needs ~2-4GB for mxbai-embed-large model
```

**Redis not available:**
```bash
# QueryMind will fall back to in-memory cache
# To use Redis, ensure it's running:
redis-cli ping
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
│   │   ├── logging_config.py # Structured logging (NEW)
│   │   ├── embeddings.py     # ChromaDB embeddings
│   │   ├── cache.py          # Query caching (Redis)
│   │   └── conversation_memory.py  # Conversation stub (NEW)
│   ├── agents/               # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base_agent.py     # Abstract base agent
│   │   ├── fast_search_agent.py    # Quick keyword search
│   │   ├── deep_research_agent.py  # LLM-powered search
│   │   ├── vault_search_agent_local.py  # Ollama integration (NEW)
│   │   ├── web_search_client.py    # Web search fallback (NEW)
│   │   └── router.py         # Intelligent routing
│   └── mcp/                  # Model Context Protocol
│       └── server.py         # FastMCP server
├── tests/                    # Test suite (NEW)
│   ├── __init__.py
│   ├── test_imports_work.py  # Import verification
│   ├── test_router_basic.py  # Routing logic tests
│   └── test_security_validation.py  # Security tests
├── pyproject.toml            # Package configuration (NEW)
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
