"""
Main entry point for PAG with A*-PO training and evaluation
"""

import argparse
import os
import sys
import torch
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from training.trainer import PAGTrainer
from training.evaluator import PAGEvaluator


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PAG with A*-PO for Mathematical Reasoning")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "both"], default="both",
                       help="Mode: train, eval, or both")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model for evaluation")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to custom config file")
    parser.add_argument("--eval_output", type=str, default=None,
                       help="Output file for evaluation results")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    if args.use_wandb:
        config.training.use_wandb = True
    
    if args.device != "auto":
        config.training.device = args.device
    elif torch.cuda.is_available():
        config.training.device = "cuda"
    else:
        config.training.device = "cpu"
    
    # Load custom config if provided
    if args.config_file:
        if args.config_file.endswith('.json'):
            # Load JSON config
            with open(args.config_file, "r") as f:
                custom_config = json.load(f)
                # Update config with custom values
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        elif args.config_file.endswith('.py'):
            # Load Python config (import it)
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_config", args.config_file)
            custom_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config_module)
            
            # Update config with custom values
            if hasattr(custom_config_module, 'config'):
                custom_config = custom_config_module.config
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        else:
            print(f"Warning: Unsupported config file format: {args.config_file}")
            print("Supported formats: .json, .py")
    
    print("="*60)
    print("PAG with A*-PO for Mathematical Reasoning")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {config.training.device}")
    print(f"Output Directory: {config.training.output_dir}")
    print(f"Model: {config.model.model_name}")
    print(f"Dataset: {config.data.dataset_name}")
    print("="*60)
    
    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.training.seed)
        torch.cuda.manual_seed_all(config.training.seed)
    
    try:
        if args.mode in ["train", "both"]:
            print("\nStarting training...")
            trainer = PAGTrainer(config)
            trainer.train()
            print("Training completed successfully!")
        
        if args.mode in ["eval", "both"]:
            print("\nStarting evaluation...")
            
            # Determine model path
            model_path = args.model_path
            if not model_path:
                # Look for best model in output directory
                best_model_path = os.path.join(config.training.output_dir, "best_model")
                if os.path.exists(best_model_path):
                    model_path = best_model_path
                else:
                    print("No model path provided and no best model found. Please specify --model_path")
                    return
            
            # Initialize evaluator
            evaluator = PAGEvaluator(config, model_path)
            
            # Run evaluation
            eval_output = args.eval_output
            if not eval_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                eval_output = os.path.join(config.training.output_dir, f"eval_results_{timestamp}.json")
            
            results = evaluator.evaluate_on_math500(eval_output)
            
            print(f"\nEvaluation completed! Results saved to {eval_output}")
            print(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_single_problem():
    """Test the system on a single problem"""
    print("\nRunning quick system test...")
    
    # Run the quick test instead of model loading
    import subprocess
    import sys
    
    try:
        result = subprocess.run([sys.executable, "quick_test.py"], 
                              capture_output=True, text=True, cwd=".")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running quick test: {e}")
        return False


if __name__ == "__main__":
    # Check if we want to run a quick test
    if len(sys.argv) == 1:
        print("Running quick test...")
        test_single_problem()
    else:
        exit_code = main()
        sys.exit(exit_code)
