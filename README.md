# Letta-hacking tools

## Prerequisites

- [Letta Cloud API key](https://docs.letta.com/guides/cloud/letta-api-key/)
- [uv](https://docs.astral.sh/uv/) or plain old [pip](https://pypi.org/project/pip/)

## Agents <=> Memory Blocks

Visualize Letta Agents and Memory Blocks as an interactive graph. Agent nodes are clickable into the Letta ADE.

### Install

```
uv venv
uv pip install -r requirements.txt
```

### Run

```
source .venv/bin/activate
python agent_memory_blocks.py -h
python agent_memory_blocks.py
```

### Example output

*Blue for agents, green for memory blocks:*

![Agent Memory Blocks](docs/agent_memory_blocks.png)
