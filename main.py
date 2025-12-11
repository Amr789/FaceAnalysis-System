import argparse
import sys
import os

# Import modules from your project structure
# We wrap imports in try-except to give helpful errors if dependencies are missing
try:
    from src.train import train
    from src.inference import UTKFacePipeline
    from scripts.setup_data import setup_kaggle_and_download
except ImportError as e:
    print(f"❌ ImportError: {e}")
    print("Ensure you are running this from the project root and have installed requirements.")
    sys.exit(1)

def run_setup():
    """Wrapper for the data setup script."""
    print("🚀 Starting Project Setup...")
    setup_kaggle_and_download()

def run_training(args):
    """Wrapper for the training loop."""
    if not os.path.exists(args.data):
        print(f"❌ Error: Dataset not found at '{args.data}'.")
        print("Did you run 'python main.py --mode setup' first?")
        sys.exit(1)
        
    print(f"🏋️‍♀️ Starting Training for {args.epochs} epochs...")
    train(
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        save_path=args.save
    )

def run_prediction(args):
    """Wrapper for inference."""
    if not args.image1 or not args.image2:
        print("❌ Error: --image1 and --image2 are required for prediction.")
        sys.exit(1)
        
    print(f"🔍 Analyzing images:\n A: {args.image1}\n B: {args.image2}")
    
    pipeline = UTKFacePipeline(age_model_path=args.save) # Re-use save path for loading
    result = pipeline.compare(args.image1, args.image2)
    
    print("\n" + "="*30)
    print("📊 ANALYSIS RESULTS")
    print("="*30)
    for key, value in result.items():
        print(f"{key:<25}: {value}")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(description="Face Analysis System CLI")
    
    # Create the 'mode' argument (setup, train, predict)
    parser.add_argument("--mode", type=str, required=True, choices=["setup", "train", "predict"],
                        help="Action to perform: 'setup' (download data), 'train' (train model), or 'predict' (compare images)")
    
    # Training Arguments
    parser.add_argument("--data", type=str, default="data/UTKFace", help="Path to dataset folder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--save", type=str, default="models/utk_age_model.pth", help="Path to save/load model weights")
    
    # Prediction Arguments
    parser.add_argument("--image1", type=str, help="Path to first image for comparison")
    parser.add_argument("--image2", type=str, help="Path to second image for comparison")

    args = parser.parse_args()

    # Route to correct function
    if args.mode == "setup":
        run_setup()
    elif args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_prediction(args)

if __name__ == "__main__":
    main()