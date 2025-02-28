
import torch
import argparse
import os
import json
from model import IterativeReasoningModel, ReasoningDataset, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Iterative Reasoning Model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_iterations", type=int, default=20, help="Number of reasoning iterations")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load the model
    model = IterativeReasoningModel()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Load the evaluation data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    dataset = ReasoningDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate the model
    results = evaluate_model(model, dataloader, num_iterations=args.num_iterations)
    
    # Save the results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()