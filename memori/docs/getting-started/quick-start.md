# Quick Start

Get Memori running in 3 minutes.

## 1. Install

```bash
pip install memorisdk litellm
```

## 2. Set API Key

```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
```

## 3. Basic Usage

Create `demo.py`:

```python
from memori import Memori
from litellm import completion

# Initialize memory
memori = Memori(conscious_ingest=True)
memori.enable()

# First conversation - establish context
response1 = completion(
    model="gpt-4o-mini",
    messages=[{
        "role": "user", 
        "content": "I'm working on a Python FastAPI project"
    }]
)
print("Assistant:", response1.choices[0].message.content)

# Second conversation - memory provides context  
response2 = completion(
    model="gpt-4o-mini", 
    messages=[{
        "role": "user",
        "content": "Help me add user authentication"
    }]
)
print("Assistant:", response2.choices[0].message.content)
```

## 4. Run

```bash
python demo.py
```

## 5. See Results

- First response: General FastAPI help
- Second response: **Contextual authentication help** (knows about your FastAPI project!)
- Database created: `memori.db` with your conversation memories

## What Happened?

1. **Universal Recording**: `memori.enable()` automatically captures ALL LLM conversations
2. **Intelligent Processing**: Extracts entities (Python, FastAPI, projects) and categorizes memories
3. **Context Injection**: Second conversation automatically includes relevant memories
4. **Persistent Storage**: All memories stored in SQLite database for future sessions

## Next Steps

- [Basic Usage](basic-usage.md) - Learn core concepts
- [Configuration](../configuration/settings.md) - Customize for your needs
- [Examples](https://github.com/GibsonAI/memori/tree/main/examples) - Explore more examples

!!! tip "Pro Tip"
    Try asking the same questions in a new session - Memori will remember your project context!