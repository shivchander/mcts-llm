# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of MCTSr (Monte Carlo Tree Search + Self-Refine) for mathematical problem solving, based on "Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B" by Zhang et al.

## Architecture

### Core Components

- **src/mcts_llm/mctsr.py**: Core MCTSr algorithm implementation
  - `MCTSNode`: Tree node representing solution attempts  
  - `MCTSr`: Base class with tree operations (selection, expansion, backpropagation)
  - `MCTSrLlama38B` & `MCTSrGPT4o`: Model-specific implementations
  - Selection policies: Greedy, Importance Sampling, Pairwise Importance Sampling
  - Initialize strategies: Zero-shot, Dummy Answer

- **src/mcts_llm/llm.py**: LLM interface and API client management
  - Supports OpenAI, Anthropic, and Fireworks APIs
  - Handles different base URLs for different providers

- **src/mcts_llm/prompt_configs.py**: Model-specific prompting configurations
  - System prompts for critique, refinement, and evaluation phases
  - Response format schemas (JSON for GPT-4o, text for Llama)

- **src/dataset_utils.py**: Dataset loading utilities
  - AIME dataset (mathematical competition problems)
  - GSM8K dataset support
  - GSM-Hard dataset integration

### Tree Search Process

1. **Initialize**: Generate initial solution (zero-shot or dummy)
2. **Select**: Choose non-fully-expanded node using UCT scores
3. **Expand**: Generate critique → refine solution → create child node
4. **Evaluate**: Score solution quality (-100 to 100 scale)
5. **Backpropagate**: Update Q-values up the tree

## Development Commands

**Package Management**: All commands use `uv` as the package manager.

```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run Python scripts
uv run python script.py
```

**Environment Setup**: Create `.env` file with API keys:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
FIREWORKS_API_KEY=your_fireworks_key
```

**Datasets**: Located in `datasets/` directory:
- `AIME_Dataset_1983_2024.csv`: Competition math problems
- `gs8mk-test.jsonl`: GSM8K test problems

## Testing

No formal test suite found. Testing appears to be done through result evaluation in `results/` directory.

## Key Design Patterns

- **Abstract Base Classes**: `MCTSr` defines interface, subclasses implement model-specific logic
- **Strategy Pattern**: Selection policies and initialization strategies are configurable
- **Pydantic Models**: Used for data validation and structured responses
- **Reward Aggregation**: Combines average and min rewards to balance optimism/pessimism

## Model Configuration

Two primary configurations:
- **Llama-3-8B**: Via Fireworks API, text-based responses
- **GPT-4o**: Via OpenAI API, JSON-structured responses with explicit thought processes

Evaluation uses strict scoring (max 95/100) with penalties for over-confidence.