# QueryMind User Guide

**How to actually use QueryMind in your daily workflow**

---

## 🎯 Three Ways to Use QueryMind

### 1. **With Claude Code** (Recommended for Claude users)
### 2. **Terminal/CLI** (For command-line power users)
### 3. **Python Scripts** (For custom integrations)

---

## 🤖 Option 1: Use with Claude Code (MCP Integration)

**Best for**: Claude Code users who want AI-powered vault search

### Setup (One-time, 2 minutes)

1. **Start QueryMind services**:
   ```bash
   cd querymind/docker
   docker-compose up -d
   ```

2. **Add MCP server to Claude Code config**:

   Edit `~/.claude/config.json`:
   ```json
   {
     "mcpServers": {
       "querymind": {
         "command": "docker",
         "args": [
           "exec",
           "querymind-mcp",
           "fastmcp",
           "run",
           "querymind.mcp.server"
         ]
       }
     }
   }
   ```

3. **Restart Claude Code**

### Daily Usage

Once configured, QueryMind tools are available in Claude Code chat:

**Example 1: Simple search**
```
You: @querymind search for "Redis caching patterns"

Claude: Let me search your vault...
[QueryMind returns 5 relevant documents about Redis caching]
```

**Example 2: Complex question (automatic deep research)**
```
You: @querymind How should I architect Redis caching for microservices?

Claude: I'll use deep research for this complex question...
[QueryMind analyzes multiple documents and synthesizes an answer]
```

**Example 3: Recent documents only**
```
You: @querymind find documents about AI from the last week

Claude: Searching with temporal filter...
[QueryMind searches only documents from Oct 17-24, 2025]
```

### Available MCP Tools

QueryMind provides 15 tools to Claude Code:

| Tool | Purpose | Example |
|------|---------|---------|
| `auto_search_vault` | Smart search with routing | "find Redis docs" |
| `web_search_vault` | External knowledge | "latest Redis 2025 features" |
| `explore_vault_structure` | Browse directories | "show me my project folders" |
| `grep_vault_content` | Fast text search | "find all TODO comments" |
| `index_file_to_chromadb` | Index new file | "index this document" |
| `get_gpu_status` | Check GPU usage | "is GPU being used?" |
| `list_ollama_models` | See available LLMs | "what models are installed?" |
| ...and 8 more | See docs/ | — |

**Key point**: You never call these directly - Claude Code uses them automatically when you ask questions!

---

## 💻 Option 2: Terminal/CLI Usage

**Best for**: Power users who want command-line control

### Quick Commands

```bash
# Start services (one time per boot)
cd querymind/docker && docker-compose up -d

# Simple search from terminal
python -m querymind.cli search "Redis caching"

# Complex question (triggers deep research)
python -m querymind.cli ask "How to implement caching?"

# Search with date filter
python -m querymind.cli search "project updates" --date 2025-10-24

# Index new documents
python -m querymind.cli index ~/Documents/vault --recursive

# Check system health
python -m querymind.cli health
```

### Terminal Output Example

```bash
$ python -m querymind.cli search "Redis caching"

🔍 Searching vault for: "Redis caching"
🧠 Router decision: FastSearch (<1s)
⚡ Cache hit: false

Results (5 found in 0.82s):

1. [0.92] 04-Knowledge-Systems/Redis-Caching-Guide.md
   Preview: Redis caching is a powerful technique for improving
   application performance by storing frequently accessed data...

2. [0.87] 02-Active-Work/API-Optimization.md
   Preview: When implementing caching with Redis, consider these
   patterns: cache-aside, write-through, and write-behind...

3. [0.81] 03-Projects/Microservices-Architecture.md
   ...

📊 Stats:
   Agent: FastSearch
   Time: 0.82s
   Cache: miss → now cached for 1hr
   Documents searched: 38,380
```

### Bash Aliases (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Quick QueryMind commands
alias qm='python -m querymind.cli'
alias qms='python -m querymind.cli search'
alias qma='python -m querymind.cli ask'

# Now you can use:
# qms "Redis caching"
# qma "How to implement Redis?"
```

---

## 🐍 Option 3: Python Scripts (Custom Integrations)

**Best for**: Developers building custom tools

### Basic Search Script

```python
#!/usr/bin/env python3
from querymind import search

# Simple search
results = search(
    query="Redis caching patterns",
    n_results=5
)

# Print results
for i, result in enumerate(results['results'], 1):
    print(f"{i}. {result['file']}")
    print(f"   Score: {result['score']:.2f}")
    print(f"   {result['content'][:100]}...\n")
```

### Intelligent Routing Script

```python
from querymind import auto_search

# Router automatically picks best agent
response = auto_search(
    query="How should I implement Redis caching?",
    verbose=True
)

print(f"Agent used: {response['agent_type']}")
print(f"Time taken: {response['elapsed_time']:.2f}s")
print(f"Results: {len(response['results'])}")
```

### Temporal Search Script

```python
from querymind import search

# Search only documents from October 2025
results = search(
    query="project updates",
    start_date="2025-10-01",
    end_date="2025-10-31"
)

print(f"Found {len(results['results'])} documents from October")
```

### Batch Indexing Script

```python
from querymind import index_vault

# Index entire vault
stats = index_vault(
    vault_path="/home/user/Documents/vault",
    recursive=True,
    batch_size=100
)

print(f"Indexed {stats['total_docs']} documents")
print(f"Time: {stats['elapsed_time']:.1f}s")
print(f"Avg: {stats['avg_per_doc']:.2f}s/doc")
```

---

## 🔒 Local-Only Mode (No External API Calls)

QueryMind can run **100% locally** without any external API calls:

### Disable Web Search (Optional)

**Option 1**: Don't set the API key
```bash
# In docker/.env, leave this empty:
SERPER_API_KEY=
```

**Option 2**: Disable web search in router
```bash
# In docker/.env, add:
DISABLE_WEB_SEARCH=true
```

### What happens without web search?

- ✅ **FastSearch**: Still works (local BM25 + vector search)
- ✅ **DeepResearch**: Still works (local Ollama LLMs)
- ❌ **WebSearch**: Disabled (queries fail with helpful message)

**Example**:
```bash
$ python -m querymind.cli search "latest Redis 2025 features"

⚠️  WebSearch is disabled (no SERPER_API_KEY)
🧠 Falling back to DeepResearch with local LLMs...

Results (3 found in 12.4s):
[Shows results from your vault only, no external search]
```

### Completely Air-Gapped Usage

QueryMind works in air-gapped environments:

1. **No internet required** for core functionality
2. **All LLMs run locally** via Ollama
3. **All embeddings local** via ChromaDB
4. **Redis cache local** (no external connections)

**Only external call**: WebSearch (optional, can be disabled)

---

## 🎨 Customization Options

### Change LLM Models

```bash
# In docker/.env, specify models to use:
OLLAMA_MODELS_TO_PULL=mistral:7b,qwen2.5-coder:14b,llama2:13b

# Restart Ollama service
docker restart querymind-ollama
```

### Adjust Cache TTLs

```bash
# In docker/.env:
CACHE_TTL_QUERY=3600     # 1 hour (default)
CACHE_TTL_GATHER=300     # 5 minutes (default)

# Longer cache = faster repeated queries
# Shorter cache = fresher results for changing vaults
```

### Change Router Threshold

```bash
# In docker/.env:
ROUTER_FAST_THRESHOLD=10  # Default: 10 words

# Lower = more queries use FastSearch (faster but less intelligent)
# Higher = more queries use DeepResearch (slower but more comprehensive)
```

---

## 📊 Understanding Performance

### When to expect <1s results (FastSearch)

✅ Simple keyword queries:
- "Redis caching"
- "machine learning pipelines"
- "kubernetes configuration"

### When to expect ~10s results (DeepResearch)

🧠 Complex questions:
- "How should I architect Redis caching?"
- "What are best practices for error handling?"
- "Explain the differences between StatefulSet and Deployment"

### When to expect ~3s results (WebSearch)

🌐 External knowledge:
- "latest Redis features 2025"
- "Anthropic Claude 3.5 Sonnet release date"
- "quantum computing news"

**Key insight**: QueryMind **automatically routes** queries to the right agent. You don't choose - the 7-heuristic router decides in <50μs.

---

## 🛠️ Troubleshooting

### "Connection refused" errors

**Cause**: Services not running

**Fix**:
```bash
cd querymind/docker
docker-compose up -d
./scripts/health-check.sh
```

### Slow searches (>5s for simple queries)

**Cause**: Cache not working or GPU not available

**Fix**:
```bash
# Check Redis cache
docker exec querymind-redis redis-cli INFO stats | grep keyspace_hits

# Check GPU usage (if available)
nvidia-smi

# Restart services
docker-compose restart
```

### "No results found" but documents exist

**Cause**: Documents not indexed

**Fix**:
```bash
# Index your vault
python -m querymind.cli index ~/Documents/vault --recursive

# Verify indexing
docker exec querymind-mcp python -c "
from querymind.core.embeddings import get_collection_stats
print(get_collection_stats())
"
```

---

## 💡 Pro Tips

### 1. Use Bash Aliases

```bash
# Add to ~/.bashrc
alias qm-start='cd ~/querymind/docker && docker-compose up -d'
alias qm-stop='cd ~/querymind/docker && docker-compose down'
alias qm-search='python -m querymind.cli search'
alias qm-health='cd ~/querymind && ./scripts/health-check.sh'
```

### 2. Index on a Schedule

```bash
# Add to crontab (runs daily at 3 AM)
0 3 * * * cd ~/querymind && python -m querymind.cli index ~/Documents/vault --recursive >> /var/log/querymind.log 2>&1
```

### 3. Monitor Cache Hit Rate

```bash
# Check cache performance
docker exec querymind-redis redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"

# Goal: >70% hit rate
```

### 4. Use Temporal Search for Recent Work

```bash
# Find what you worked on this week
python -m querymind.cli search "project updates" \
  --start-date "$(date -d '7 days ago' +%Y-%m-%d)" \
  --end-date "$(date +%Y-%m-%d)"
```

---

## 🔗 Related Documentation

- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup guide
- [ARCHITECTURE.md](docs/architecture.md) - Technical deep-dive
- [MCP Integration](docs/mcp-integration.md) - Full MCP setup guide
- [Performance Tuning](docs/performance.md) - Optimization tips

---

**Questions?** See [docs/troubleshooting.md](docs/troubleshooting.md) or open a [GitHub issue](https://github.com/querymind/querymind/issues)
