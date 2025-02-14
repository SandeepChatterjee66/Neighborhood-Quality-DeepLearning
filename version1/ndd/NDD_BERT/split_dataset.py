import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and split dataset')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='Path to input CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    try:
        # Read input file
        df = pd.read_csv(args.input_path)
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Create output directory path
        output_dir = os.path.dirname(args.input_path)
        
        # Create sample.csv with 1024 rows
        sample_df = df.sample(n=min(1024, len(df)), random_state=args.seed)
        sample_path = os.path.join(output_dir, 'sample.csv')
        sample_df.to_csv(sample_path, index=False)
        print(f"Saved sample.csv with {len(sample_df)} rows")
        
        # Split remaining data into train/validation
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=args.seed
        )
        
        # Save train and validation files
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'val.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"Split completed successfully:")
        print(f"- Train set: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
        print(f"- Validation set: {len(val_df)} rows ({len(val_df)/len(df):.1%})")
        print(f"Files saved in: {output_dir}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_path}")
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty or corrupted")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    main()