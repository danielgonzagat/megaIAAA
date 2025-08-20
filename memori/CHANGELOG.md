# Changelog

All notable changes to Memori will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-08-03

### 🚀 **Dual-Mode Memory System - Revolutionary Architecture**

**Major Release**: Complete overhaul of memory injection system with two distinct modes - Conscious short-term memory and Auto dynamic search.

#### ✨ **New Memory Modes**

**🧠 Conscious Mode (`conscious_ingest=True`)**
- **Short-Term Working Memory**: Mimics human conscious memory with essential info readily available
- **Startup Analysis**: Conscious agent analyzes long-term memory patterns at initialization
- **Memory Promotion**: Automatically promotes 5-10 essential conversations from long-term to short-term storage
- **One-Shot Injection**: Injects working memory context ONCE at conversation start, no repetition
- **Essential Context**: Names, current projects, preferences, skills always accessible

**🔍 Auto Mode (`auto_ingest=True`)**
- **Dynamic Database Search**: Uses retrieval agent for intelligent full-database search
- **Query Analysis**: AI-powered query understanding with OpenAI Structured Outputs
- **Continuous Retrieval**: Searches and injects 3-5 relevant memories on EVERY LLM call
- **Performance Optimized**: Caching, async processing, background threading
- **Full Coverage**: Searches both short-term and long-term memory databases

**⚡ Combined Mode (`conscious_ingest=True, auto_ingest=True`)**
- **Best of Both Worlds**: Working memory foundation + dynamic search capability
- **Layered Context**: Essential memories + query-specific memories
- **Maximum Intelligence**: Comprehensive memory utilization

#### 🔧 **API Changes**

**New Parameters**
```python
memori = Memori(
    conscious_ingest=True,  # Short-term working memory (one-shot)
    auto_ingest=True,       # Dynamic database search (continuous)
    openai_api_key="sk-..."
)
```

**Mode Behaviors**
- **Conscious**: Analysis at startup → Memory promotion → One-shot context injection
- **Auto**: Query analysis → Database search → Context injection per call
- **Combined**: Startup analysis + Per-call search

#### 🏗️ **Architecture Improvements**

**Enhanced Agents**
- **Conscious Agent**: Smarter long-term → short-term memory promotion
- **Retrieval Agent**: Performance optimized with caching and async support
- **Memory Agent**: Improved Pydantic-based processing

**Performance Enhancements**
- **Query Caching**: 5-minute TTL cache for search plans to reduce API calls
- **Async Processing**: `execute_search_async()` for concurrent operations
- **Background Threading**: Non-blocking search execution
- **Thread Safety**: Proper locking mechanisms for concurrent access

#### 📚 **Documentation & Examples**

**Updated Examples**
- **`memori_example.py`**: Complete conscious-ingest demonstration with detailed comments
- **`auto_ingest_example.py`**: New example showcasing dynamic memory retrieval
- **Enhanced Comments**: Detailed explanations of each mode's behavior

**Updated Documentation**
- **README.md**: Comprehensive dual-mode system explanation
- **Mode Comparisons**: Clear distinctions between conscious vs auto modes
- **Configuration Examples**: All possible mode combinations

#### 🎯 **Use Cases**

**Conscious Mode Perfect For:**
- Personal assistants needing user context
- Project-specific conversations requiring background knowledge
- Situations where essential info should always be available
- One-time context establishment scenarios

**Auto Mode Perfect For:**
- Dynamic Q&A systems
- Research assistants requiring specific memory retrieval
- Multi-topic conversations needing relevant context injection
- Performance-critical applications with intelligent caching

**Combined Mode Perfect For:**
- Comprehensive personal AI assistants
- Maximum context utilization scenarios
- Professional applications requiring both background and specific context

#### 🛠️ **Developer Experience**

**Simplified Configuration**
```json
{
  "agents": {
    "conscious_ingest": true,
    "auto_ingest": false,
    "openai_api_key": "sk-..."
  }
}
```

**Enhanced Logging**
- Detailed mode-specific logging
- Performance metrics for caching and search
- Background processing status updates

#### ⚡ **Breaking Changes**

**Behavioral Changes**
- `conscious_ingest=True` now works differently (one-shot vs continuous)
- Memory injection timing changed based on selected mode
- Context injection strategies optimized per mode

**New Dependencies**
- Enhanced async processing requirements
- Additional threading support for background operations

## [1.1.0] - 2025-08-03

### 🧠 **Enhanced Conscious Ingestion System**

Major improvements to the intelligent memory processing and context injection system.

#### ✨ New Features

**Conscious Agent System**
- **Background Analysis**: Automatic analysis of long-term memory patterns every 6 hours
- **Essential Memory Promotion**: Promotes key personal facts to short-term memory for immediate access
- **Intelligent Context Selection**: AI-powered identification of most relevant memories for context injection
- **Personal Identity Extraction**: Automatically identifies and prioritizes user identity, preferences, and ongoing projects

**Enhanced Context Injection**
- **Essential Conversations**: Priority context from promoted memories for immediate relevance
- **Smart Memory Retrieval**: Up to 5 most relevant memories automatically injected into conversations
- **Category-Aware Context**: Different context strategies for facts, preferences, skills, and rules
- **Reduced Token Usage**: More efficient context injection with summarized essential information

**Improved Memory Processing**
- **Pydantic-Based Agents**: Structured memory processing with OpenAI Structured Outputs
- **Multi-Dimensional Scoring**: Frequency, recency, and importance scoring for memory selection
- **Entity Relationship Mapping**: Enhanced entity extraction and relationship tracking
- **Advanced Categorization**: Improved classification of facts, preferences, skills, context, and rules

#### 🔧 API Enhancements

**Conscious Ingestion Control**
```python
memori = Memori(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,  # Enable intelligent background analysis
    openai_api_key="sk-..."
)
```

**Memory Retrieval Methods**
- `get_essential_conversations()` - Access promoted essential memories
- `trigger_conscious_analysis()` - Manually trigger background analysis
- `retrieve_context()` - Enhanced context retrieval with essential memory priority

#### 📊 Background Processing

**Conscious Agent Features**
- **Automated Analysis**: Runs every 6 hours to analyze memory patterns
- **Selection Criteria**: Personal identity, preferences, skills, current projects, relationships
- **Memory Promotion**: Automatically promotes essential conversations to short-term memory
- **Analysis Reasoning**: Detailed reasoning for memory selection decisions

#### 🎯 Context Injection Improvements

**Essential Memory Integration**
- Essential conversations always included in context
- Smart memory limit management (3 essential + 2 specific)
- Category-based context prioritization
- Improved relevance scoring for memory selection

#### 🛠️ Developer Experience

**Enhanced Examples**
- Updated `memori_example.py` with conscious ingestion showcase
- New `memory_retrieval_example.py` demonstrating function calling integration
- Advanced configuration examples with conscious agent settings

## [1.0.0] - 2025-08-03

### 🎉 **Production-Ready Memory Layer for AI Agents**

Complete professional-grade memory system with modular architecture, comprehensive error handling, and configuration management.

### ✨ Core Features
- **Universal LLM Integration**: Works with ANY LLM library (LiteLLM, OpenAI, Anthropic)
- **Pydantic-based Intelligence**: Structured memory processing with validation
- **Automatic Context Injection**: Relevant memories automatically added to conversations
- **Multiple Memory Types**: Short-term, long-term, rules, and entity relationships
- **Advanced Search**: Full-text search with semantic ranking

### 🏗️ Architecture
- **Modular Design**: Separated concerns with clear component boundaries
- **SQL Query Centralization**: Dedicated query modules for maintainability
- **Configuration Management**: Pydantic-based settings with auto-loading
- **Comprehensive Error Handling**: Context-aware exceptions with sanitized logging
- **Production Logging**: Structured logging with multiple output targets

### 🗄️ Database Support
- **Multi-Database**: SQLite, PostgreSQL, MySQL connectors
- **Query Optimization**: Indexed searches and connection pooling
- **Schema Management**: Version-controlled migrations and templates
- **Full-Text Search**: FTS5 support for advanced text search

### 🔧 Developer Experience
- **Type Safety**: Full Pydantic validation throughout
- **Simple API**: One-line enablement with `memori.enable()`
- **Flexible Configuration**: File, environment, or programmatic setup
- **Rich Examples**: Basic usage, personal assistant, advanced config

### 📊 Memory Processing
- **Entity Extraction**: People, technologies, projects, skills
- **Smart Categorization**: Facts, preferences, skills, rules, context
- **Importance Scoring**: Multi-dimensional relevance assessment
- **Relationship Mapping**: Entity interconnections and memory graphs

### 🔌 Integrations
- **LiteLLM Native**: Uses official callback system (recommended)
- **OpenAI/Anthropic**: Clean wrapper classes for direct usage
- **Tool Support**: Memory search tools for function calling

### 🛡️ Security & Reliability
- **Input Sanitization**: Protection against injection attacks
- **Error Context**: Detailed error information without exposing secrets
- **Graceful Degradation**: Continues operation when components fail
- **Resource Management**: Automatic cleanup and connection pooling

### 📁 Project Structure
```
memori/
├── core/              # Main memory interface and database
├── config/            # Configuration management system
├── agents/            # Pydantic-based memory processing
├── database/          # Multi-database support and queries
├── integrations/      # LLM provider integrations
├── utils/             # Helpers, validation, logging
└── tools/             # Memory search and retrieval tools
```

### 🎯 Philosophy Alignment
- **Second-memory for LLM work**: Never repeat context again
- **Flexible database connections**: Production-ready adapters
- **Simple, reliable architecture**: Just works out of the box
- **Conscious context injection**: Intelligent memory retrieval

### ⚡ Quick Start
```python
from memori import Memori

memori = Memori(
    database_connect="sqlite:///my_memory.db",
    conscious_ingest=True,
    openai_api_key="sk-..."
)
memori.enable()  # Start recording all LLM conversations

# Use any LLM library - context automatically injected!
from litellm import completion
response = completion(model="gpt-4", messages=[...])
```

### 📚 Documentation
- Clean, focused README aligned with project vision
- Essential examples without complexity bloat
- Configuration guides for development and production
- Architecture documentation for contributors

### 🗂️ Archive Management
- Moved outdated files to `archive/` folder
- Updated `.gitignore` to exclude archive from version control
- Preserved development history while cleaning main structure

### 💡 Breaking Changes from Pre-1.0
- Moved from enum-driven to Pydantic-based processing
- Simplified API surface with focus on `enable()/disable()`
- Restructured package layout for better modularity
- Enhanced configuration system replaces simple parameters

---

*This release represents the culmination of the vision outlined in the original architecture documents, delivering a production-ready memory layer that "just works" for AI developers.*