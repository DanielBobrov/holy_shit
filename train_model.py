import torch
import torch.nn as nn
import argparse
import os
import json
from model import IterativeReasoningModel, ReasoningDataset, train_model, evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Iterative Reasoning Model")
    
    parser.add_argument("--vocab_path", type=str, default="/home/user/Pycharm/shit/reasoning/vocab.json",
                        help="Path to vocabulary file")
    parser.add_argument("--train_path", type=str, default="/home/user/Pycharm/shit/reasoning/train.json",
                        help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="/home/user/Pycharm/shit/reasoning/valid.json",
                        help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="/home/user/Pycharm/shit/reasoning/test.json",
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model and results")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Dimension of feed-forward network")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of layers in thinking block")
    parser.add_argument("--max_iterations", type=int, default=20,
                        help="Maximum number of reasoning iterations")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading datasets...")
    
    # Load datasets
    train_dataset = ReasoningDataset(args.train_path, args.vocab_path)
    val_dataset = ReasoningDataset(args.valid_path, args.vocab_path)
    test_dataset = ReasoningDataset(args.test_path, args.vocab_path)
    
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = IterativeReasoningModel(
        vocab_size=len(train_dataset.vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_iterations=args.max_iterations,
        dropout=args.dropout
    )
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Training model...")
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
