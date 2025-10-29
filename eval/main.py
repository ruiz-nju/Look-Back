import argparse
import json
import os
import torch
from vllm import LLM, SamplingParams
from utils.data_loaders import (
    load_wemath_dataset,
    load_mathvista_dataset,
    load_mathverse_dataset,
    load_mathvision_dataset,
    load_hallubench_dataset,
    load_GeoMath_dataset,
    load_Tallyqa_dataset,
    load_MME_dataset,
)
from utils.processing import (
    prepare_prompts,
    process_outputs_simplified,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for multimodal math datasets"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument("--min-pixels", type=int, default=262144)
    parser.add_argument("--max-pixels", type=int, default=1000000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.",
        help="System prompt for the model",
    )
    parser.add_argument("--version", type=str, default="back")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated list of datasets to evaluate: geo3k,wemath,mathvista,mathverse,mathvision or 'all'",
    )
    parser.add_argument("--data-path", type=str, default="eval/eval_data", help="")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which datasets to evaluate
    datasets_to_eval = (
        args.datasets.split(",")
        if args.datasets != "all"
        else [
            "wemath",
            "mathvista",
            "mathverse",
            "mathvision",
            "hallubench",
            "GeoMath",
            "Tallyqa",
            "MME",
        ]
    )

    # Dictionary to store all samples
    all_samples = {}

    # Load datasets based on selection
    for dataset_name in datasets_to_eval:
        if dataset_name == "wemath":
            all_samples["wemath"] = load_wemath_dataset(args.data_path)
            print(f"Loaded {len(all_samples['wemath'])} samples from WeMath")

        elif dataset_name == "mathvista":
            all_samples["mathvista"] = load_mathvista_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathvista'])} samples from MathVista")

        elif dataset_name == "mathverse":
            all_samples["mathverse"] = load_mathverse_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathverse'])} samples from MathVerse")

        elif dataset_name == "mathvision":
            all_samples["mathvision"] = load_mathvision_dataset(args.data_path)
            print(f"Loaded {len(all_samples['mathvision'])} samples from MathVision")

        elif dataset_name == "hallubench":
            all_samples["hallubench"] = load_hallubench_dataset(args.data_path)
            print(f"Loaded {len(all_samples['hallubench'])} samples from HalluBench")

        elif dataset_name == "GeoMath":
            all_samples["GeoMath"] = load_GeoMath_dataset(args.data_path)
            print(f"Loaded {len(all_samples['GeoMath'])} samples from GeoMath")

        elif dataset_name == "Tallyqa":
            all_samples["Tallyqa"] = load_Tallyqa_dataset(args.data_path)
            print(f"Loaded {len(all_samples['Tallyqa'])} samples from Tallyqa")

        elif dataset_name == "MME":
            all_samples["MME"] = load_MME_dataset(args.data_path)
            print(f"Loaded {len(all_samples['MME'])} samples from MME")

    if not all_samples:
        print("No datasets loaded. Please check the paths and dataset names.")
        return

    # Initialize model
    print(f"Initializing model from {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.7,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # Process in batches
    all_results = {}
    for dataset_name in all_samples.keys():
        all_results[dataset_name] = []

    for dataset_name, samples in all_samples.items():
        prompts, metadata = prepare_prompts(dataset_name, samples, args)
        outputs = llm.generate(prompts, sampling_params)
        results = process_outputs_simplified(outputs, metadata)
        output_dict = {"results": results, "config": vars(args)}

        output_path = os.path.join(args.output_dir, f"{dataset_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
