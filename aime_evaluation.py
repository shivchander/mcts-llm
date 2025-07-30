#!/usr/bin/env python3
"""
Generic script to run MCTSr inference on AIME 2024 problems.
Can be used with any model and port configuration.

Usage:
    python aime_evaluation.py --model "Qwen/Qwen2.5-1.5B-Instruct" --port 8000
    python aime_evaluation.py --model "Qwen/Qwen2.5-Math-1.5B-Instruct" --port 8001
    python aime_evaluation.py --model "Qwen/Qwen2.5-Math-7B-Instruct" --port 8002
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.mcts_llm.mctsr import MCTSr, SelectionPolicy, InitializeStrategy, MCTSNode
from src.mcts_llm.llm import openai_chat_completion
from src.mcts_llm.prompt_configs import PromptConfig


def create_generic_mctsr(model_name: str, base_url: str, **kwargs):
    """Factory function to create MCTSr with dynamic configuration."""
    
    # Create dynamic prompt config
    prompt_config = PromptConfig(
        base_url=base_url,
        model=model_name,
        critic_system_prompt="Provide a detailed and constructive critique to improve the answer. "
        "Highlight specific areas that need refinement or correction.",
        refine_system_prompt="""# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
- Respond with only the answer.
- Please reason step by step, and put your final answer within \\boxed{}.
""",
        evaluate_system_prompt=(
            "Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
            "Do not give a full score above 95. Make sure the reward score is an integer. "
            "Return *ONLY* the score."
        ),
    )
    
    class DynamicMCTSr(MCTSr):
        def zero_shot(self) -> str:
            response = openai_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {
                        "role": "user",
                        "content": f"<problem>\n{self.problem}\n</problem>",
                    },
                ],
                model=prompt_config.model,
                base_url=prompt_config.base_url,
                max_tokens=4000,
            )
            assert response.choices[0].message.content is not None
            return response.choices[0].message.content

        def self_refine(self, node: MCTSNode) -> MCTSNode:
            critique_response = openai_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": prompt_config.critic_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"<problem>\n{self.problem}\n</problem>",
                                f"<current_answer>\n{node.answer}\n</current_answer>",
                            ]
                        ),
                    },
                ],
                model=prompt_config.model,
                base_url=prompt_config.base_url,
                max_tokens=4000,
            )
            critique = critique_response.choices[0].message.content
            assert critique is not None
            self.critiques.append(critique)

            refined_answer_response = openai_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": prompt_config.refine_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [
                                f"<problem>\n{self.problem}\n</problem>",
                                f"<current_answer>\n{node.answer}\n</current_answer>",
                                f"<critique>\n{critique}\n</critique>",
                            ]
                        ),
                    },
                ],
                model=prompt_config.model,
                base_url=prompt_config.base_url,
                max_tokens=4000,
            )
            refined_answer = refined_answer_response.choices[0].message.content
            assert refined_answer is not None
            self.refinements.append(refined_answer)

            return MCTSNode(answer=refined_answer, parent=node)

        def _evaluate_answer(self, node: MCTSNode) -> int:
            messages = [
                {
                    "role": "system",
                    "content": prompt_config.evaluate_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<answer>\n{node.answer}\n</answer>",
                        ]
                    ),
                },
            ]
            for attempt in range(3):
                try:
                    response = openai_chat_completion(
                        messages=messages,
                        model=prompt_config.model,
                        base_url=prompt_config.base_url,
                        max_tokens=4000,
                    )
                    assert response.choices[0].message.content is not None
                    return int(response.choices[0].message.content)
                except ValueError:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            },
                            {
                                "role": "user",
                                "content": "Failed to parse reward as an integer.",
                            },
                        ]
                    )
                    if attempt == 2:
                        # Return a default score if parsing fails
                        return 50
    
    return DynamicMCTSr(**kwargs)


def load_aime_2024() -> pd.DataFrame:
    """Load AIME 2024 dataset from HuggingFace."""
    print("Loading AIME 2024 dataset...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} problems")
    return df


def run_mcts_on_problem(
    problem: str,
    model_name: str,
    base_url: str,
    max_rollouts: int = 8,
    max_children: int = 2,
) -> Dict[str, Any]:
    """Run MCTSr on a single problem and return results."""
    
    mcts = create_generic_mctsr(
        model_name=model_name,
        base_url=base_url,
        problem=problem,
        max_rollouts=max_rollouts,
        exploration_constant=1.0,
        max_children=max_children,
        selection_policy=SelectionPolicy.IMPORTANCE_SAMPLING,
        initialize_strategy=InitializeStrategy.ZERO_SHOT,
    )
    
    start_time = time.time()
    try:
        best_answer = mcts.run()
        end_time = time.time()
        
        return {
            "success": True,
            "best_answer": best_answer,
            "root_q_value": mcts.root.Q,
            "num_critiques": len(mcts.critiques),
            "num_refinements": len(mcts.refinements),
            "inference_time": end_time - start_time,
            "error": None,
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "best_answer": None,
            "root_q_value": None,
            "num_critiques": 0,
            "num_refinements": 0,
            "inference_time": end_time - start_time,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run MCTSr on AIME 2024 problems")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--port", type=int, required=True, help="Port number (e.g., 8000)")
    parser.add_argument("--max-rollouts", type=int, default=8, help="Maximum number of rollouts")
    parser.add_argument("--max-children", type=int, default=2, help="Maximum children per node")
    parser.add_argument("--output-dir", default="results/aime_2024", help="Output directory")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting problem index")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending problem index")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create base URL
    base_url = f"http://localhost:{args.port}/v1"
    
    # Load dataset
    df = load_aime_2024()
    
    # Filter dataset if indices specified
    if args.end_idx is not None:
        df = df.iloc[args.start_idx:args.end_idx]
    else:
        df = df.iloc[args.start_idx:]
    
    # Create output filename
    model_safe_name = args.model.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_safe_name}_port{args.port}_{timestamp}.json"
    
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Problems to solve: {len(df)}")
    print(f"Max rollouts: {args.max_rollouts}")
    print(f"Output file: {output_file}")
    print("="*50)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Solving problems"):
        print(f"\nProblem {idx + 1}/{len(df)} (ID: {row['id']})")
        print(f"Problem: {row['problem'][:100]}...")
        
        result = run_mcts_on_problem(
            problem=row['problem'],
            model_name=args.model,
            base_url=base_url,
            max_rollouts=args.max_rollouts,
            max_children=args.max_children,
        )
        
        # Add problem metadata
        result.update({
            "problem_id": row['id'],
            "problem": row['problem'],
            "correct_answer": row['answer'],
            "model": args.model,
            "port": args.port,
            "timestamp": datetime.now().isoformat(),
        })
        
        results.append(result)
        
        if result["success"]:
            print(f"✓ Solved in {result['inference_time']:.1f}s, Q-value: {result['root_q_value']:.2f}")
            print(f"  Answer: {result['best_answer'][:100]}...")
        else:
            print(f"✗ Failed: {result['error']}")
        
        # Save intermediate results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Results saved to {output_file}")
    print(f"Success rate: {sum(1 for r in results if r['success'])}/{len(results)}")


if __name__ == "__main__":
    main()