#!/usr/bin/env python3
"""
Simple script to run MCTSr inference using a local Qwen model.
Requires the model to be running on localhost:8000 with OpenAI-compatible API.
"""

from src.mcts_llm.mctsr import MCTSrQwen, SelectionPolicy, InitializeStrategy


def main():
    # Example math problem
    problem = """
    A rectangle has a length of 12 cm and a width of 8 cm. 
    What is the area of the rectangle?
    """
    
    # Configure MCTSr with reasonable parameters for a small model
    mcts = MCTSrQwen(
        problem=problem.strip(),
        max_rollouts=4,  # Reduced for faster inference
        exploration_constant=1.0,
        max_children=2,
        selection_policy=SelectionPolicy.IMPORTANCE_SAMPLING,
        initialize_strategy=InitializeStrategy.ZERO_SHOT,
    )
    
    print("Problem:")
    print(problem.strip())
    print("\n" + "="*50)
    print("Running MCTSr inference...")
    print("="*50)
    
    try:
        # Run the search
        best_answer = mcts.run()
        
        print(f"\nBest answer found:")
        print(best_answer)
        print(f"\nTree structure:")
        mcts.print()
        
        print(f"\nSearch statistics:")
        print(f"- Total rollouts: {mcts.max_rollouts}")
        print(f"- Critiques generated: {len(mcts.critiques)}")
        print(f"- Refinements made: {len(mcts.refinements)}")
        print(f"- Root node Q-value: {mcts.root.Q:.2f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure your Qwen model is running on localhost:8000")
        print("Example command: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000")


if __name__ == "__main__":
    main()