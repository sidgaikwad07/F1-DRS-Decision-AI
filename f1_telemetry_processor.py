"""
Created on Sun Jun 23 22:18:52 2025

@author: sid

Complete Fixed F1 Telemetry Processor
"""
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class F1TelemetryProcessor:
    """
    A comprehensive class for processing F1 telemetry data for DRS decision AI training
    """
    
    def __init__(self, base_folder: str):
        """
        Initialize the processor with the base folder containing train/test/validation data
        
        Args:
            base_folder: Path to transformer_austria_training folder
        """
        self.base_folder = Path(base_folder)
        self.train_folder = self.base_folder / "train"
        self.test_folder = self.base_folder / "test" 
        self.validation_folder = self.base_folder / "validation"
        
        # Store loaded data
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        
    def load_json_file(self, file_path: str) -> Dict:
        """Load a single JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def debug_data_structure(self, max_files: int = 3):
        """
        Debug method to understand the structure of your JSON files
        
        Args:
            max_files: Maximum number of files to examine from each folder
        """
        print("=== DEBUGGING DATA STRUCTURE ===")
        
        for folder_name, folder_path in [
            ("train", self.train_folder),
            ("test", self.test_folder), 
            ("validation", self.validation_folder)
        ]:
            if not folder_path.exists():
                print(f"\n--- {folder_name.upper()} FOLDER NOT FOUND ---")
                continue
                
            print(f"\n--- {folder_name.upper()} FOLDER ---")
            json_files = list(folder_path.glob("*.json"))[:max_files]
            
            if not json_files:
                print("  No JSON files found")
                continue
            
            for json_file in json_files:
                print(f"\nExamining: {json_file.name}")
                data = self.load_json_file(json_file)
                
                if data is None:
                    print("  Failed to load")
                    continue
                    
                print(f"  Root type: {type(data)}")
                
                if isinstance(data, list):
                    print(f"  List length: {len(data)}")
                    if len(data) > 0:
                        print(f"  First item type: {type(data[0])}")
                        if isinstance(data[0], dict):
                            print(f"  First item keys: {list(data[0].keys())}")
                            
                elif isinstance(data, dict):
                    print(f"  Dict keys: {list(data.keys())}")
                    if 'telemetry_sequence' in data:
                        tel_seq = data['telemetry_sequence']
                        print(f"  Telemetry sequence type: {type(tel_seq)}")
                        if isinstance(tel_seq, list):
                            print(f"  Telemetry sequence length: {len(tel_seq)}")
                            if len(tel_seq) > 0:
                                print(f"  First timestep type: {type(tel_seq[0])}")
                                if isinstance(tel_seq[0], dict):
                                    print(f"  First timestep keys: {list(tel_seq[0].keys())}")
                                else:
                                    print(f"  First timestep value: {tel_seq[0]}")
                    else:
                        # Check if the dict itself looks like telemetry data
                        if any(key in data for key in ['speed', 'throttle', 'brake']):
                            print("  Dict appears to be telemetry data itself")
                        else:
                            print("  Dict doesn't contain expected telemetry fields")
    
    def _validate_sequence(self, sequence: Dict, source_name: str) -> bool:
        """
        Validate that a sequence has the expected structure
        
        Args:
            sequence: Sequence dictionary to validate
            source_name: Name of source for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a dictionary
            if not isinstance(sequence, dict):
                print(f"Warning: {source_name} - sequence is not a dictionary")
                return False
            
            # Check for telemetry_sequence key
            if 'telemetry_sequence' in sequence:
                telemetry_data = sequence['telemetry_sequence']
            else:
                # Maybe the sequence itself contains the telemetry data
                telemetry_data = sequence
            
            # Validate telemetry data
            if not isinstance(telemetry_data, list):
                print(f"Warning: {source_name} - telemetry data is not a list")
                return False
            
            if len(telemetry_data) == 0:
                print(f"Warning: {source_name} - empty telemetry sequence")
                return False
            
            # Check first timestep structure
            first_timestep = telemetry_data[0]
            if not isinstance(first_timestep, dict):
                print(f"Warning: {source_name} - timestep is not a dictionary")
                return False
            
            # Check for required fields
            required_fields = ['speed', 'throttle', 'brake']  # Minimum required fields
            missing_fields = [field for field in required_fields if field not in first_timestep]
            if missing_fields:
                print(f"Warning: {source_name} - missing required fields: {missing_fields}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating {source_name}: {e}")
            return False
    
    def load_all_sequences(self, folder_path: Path) -> List[Dict]:
        """
        Load all JSON sequence files from a folder
        
        Args:
            folder_path: Path to folder containing JSON files
            
        Returns:
            List of telemetry sequences
        """
        sequences = []
        
        if not folder_path.exists():
            print(f"Folder {folder_path} does not exist!")
            return sequences
            
        # Load individual JSON files
        for json_file in folder_path.glob("*.json"):
            print(f"Loading {json_file.name}...")
            data = self.load_json_file(json_file)
            if data:
                try:
                    # Handle different JSON structures
                    if isinstance(data, list):
                        # Validate each sequence in the list
                        valid_count = 0
                        for i, seq in enumerate(data):
                            if self._validate_sequence(seq, f"{json_file.name}[{i}]"):
                                sequences.append(seq)
                                valid_count += 1
                        print(f"  Added {valid_count}/{len(data)} valid sequences")
                        
                    elif isinstance(data, dict):
                        if 'telemetry_sequence' in data:
                            # Single sequence file like your sample
                            if self._validate_sequence(data, json_file.name):
                                sequences.append(data)
                                print(f"  Added 1 sequence")
                        else:
                            # Check if this dict itself is a sequence
                            if self._validate_sequence(data, json_file.name):
                                sequences.append(data)
                                print(f"  Added 1 sequence")
                except Exception as e:
                    print(f"Error processing {json_file.name}: {e}")
                    continue
                        
        print(f"Loaded {len(sequences)} valid sequences from {folder_path}")
        return sequences
    
    def load_all_data(self):
        """Load all training, testing, and validation data"""
        print("Loading F1 telemetry data...")
        
        self.train_data = self.load_all_sequences(self.train_folder)
        self.test_data = self.load_all_sequences(self.test_folder)
        self.validation_data = self.load_all_sequences(self.validation_folder)
        
        print(f"\nTotal sequences loaded:")
        print(f"  Train: {len(self.train_data) if self.train_data else 0}")
        print(f"  Test: {len(self.test_data) if self.test_data else 0}")
        print(f"  Validation: {len(self.validation_data) if self.validation_data else 0}")
    
    def sequence_to_dataframe(self, sequence: Dict) -> pd.DataFrame:
        """
        Convert a single telemetry sequence to pandas DataFrame
        
        Args:
            sequence: Single telemetry sequence dictionary
            
        Returns:
            DataFrame with telemetry data
        """
        try:
            # Extract telemetry data
            if 'telemetry_sequence' in sequence:
                telemetry_data = sequence['telemetry_sequence']
            else:
                # Assume the sequence itself is the telemetry data
                telemetry_data = sequence
            
            # Handle case where telemetry_data might not be a list
            if not isinstance(telemetry_data, list):
                print(f"Warning: telemetry_data is not a list, type: {type(telemetry_data)}")
                return pd.DataFrame()
            
            if len(telemetry_data) == 0:
                print("Warning: empty telemetry sequence")
                return pd.DataFrame()
            
            # Check if all elements are dictionaries
            if not all(isinstance(item, dict) for item in telemetry_data):
                print("Warning: not all telemetry items are dictionaries")
                # Filter to only keep dictionary items
                telemetry_data = [item for item in telemetry_data if isinstance(item, dict)]
                if len(telemetry_data) == 0:
                    return pd.DataFrame()
            
            # Create DataFrame with error handling
            df = pd.DataFrame(telemetry_data)
            
            # Add metadata columns if available
            for key in ['original_sequence_id', 'year', 'driver', 'dataset_source']:
                if key in sequence:
                    df[key] = sequence[key]
                    
            return df
            
        except Exception as e:
            print(f"Error converting sequence to dataframe: {e}")
            print(f"Sequence keys: {sequence.keys() if isinstance(sequence, dict) else 'Not a dict'}")
            if isinstance(sequence, dict) and 'telemetry_sequence' in sequence:
                tel_seq = sequence['telemetry_sequence']
                print(f"Telemetry sequence type: {type(tel_seq)}")
                if isinstance(tel_seq, list) and len(tel_seq) > 0:
                    print(f"First item type: {type(tel_seq[0])}")
                    print(f"First item: {tel_seq[0]}")
            return pd.DataFrame()
    
    def create_combined_dataframe(self, sequences: List[Dict], split_name: str = "") -> pd.DataFrame:
        """
        Combine all sequences into a single DataFrame
        
        Args:
            sequences: List of telemetry sequences
            split_name: Name of the data split (train/test/validation)
            
        Returns:
            Combined DataFrame
        """
        if not sequences:
            return pd.DataFrame()
            
        all_dfs = []
        failed_sequences = 0
        
        for i, sequence in enumerate(sequences):
            df = self.sequence_to_dataframe(sequence)
            if not df.empty:
                df['sequence_index'] = i
                df['split'] = split_name
                all_dfs.append(df)
            else:
                failed_sequences += 1
                if failed_sequences <= 5:  # Only show first 5 failures
                    print(f"Failed to convert sequence {i} in {split_name}")
        
        if failed_sequences > 0:
            print(f"Warning: {failed_sequences} sequences failed to convert in {split_name}")
        
        if not all_dfs:
            print(f"No valid sequences found in {split_name}")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    
    def analyze_data_distribution(self):
        """Analyze the distribution of telemetry data"""
        if not self.train_data:
            print("No training data loaded. Run load_all_data() first.")
            return
            
        # Create combined DataFrames
        print("Creating combined DataFrames...")
        train_df = self.create_combined_dataframe(self.train_data, "train")
        test_df = self.create_combined_dataframe(self.test_data, "test")
        val_df = self.create_combined_dataframe(self.validation_data, "validation")
        
        # Combine all data
        all_dfs = [df for df in [train_df, test_df, val_df] if not df.empty]
        if not all_dfs:
            print("No valid data found for analysis")
            return pd.DataFrame()
        
        all_df = pd.concat(all_dfs, ignore_index=True)
        
        print("\n=== DATA DISTRIBUTION ANALYSIS ===")
        print(f"Total data points: {len(all_df)}")
        print(f"Number of unique sequences: {all_df['sequence_index'].nunique()}")
        print(f"Number of unique drivers: {all_df['driver'].nunique() if 'driver' in all_df.columns else 'N/A'}")
        
        # Telemetry statistics
        telemetry_cols = ['speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm']
        available_cols = [col for col in telemetry_cols if col in all_df.columns]
        
        print(f"\n=== TELEMETRY STATISTICS ===")
        if available_cols:
            print(all_df[available_cols].describe())
        else:
            print("No telemetry columns found!")
        
        return all_df
    
    def visualize_sample_sequence(self, sequence_idx: int = 0, split: str = "train"):
        """
        Visualize a sample telemetry sequence
        
        Args:
            sequence_idx: Index of sequence to visualize
            split: Which data split to use (train/test/validation)
        """
        # Get the appropriate data
        if split == "train" and self.train_data:
            data = self.train_data
        elif split == "test" and self.test_data:
            data = self.test_data
        elif split == "validation" and self.validation_data:
            data = self.validation_data
        else:
            print(f"No {split} data available")
            return
            
        if sequence_idx >= len(data):
            print(f"Sequence index {sequence_idx} out of range. Max: {len(data)-1}")
            return
            
        # Convert to DataFrame
        df = self.sequence_to_dataframe(data[sequence_idx])
        
        if df.empty:
            print(f"Could not convert sequence {sequence_idx} to DataFrame")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'F1 Telemetry Sequence {sequence_idx} ({split})', fontsize=16)
        
        # Check which columns are available
        cols_to_plot = ['speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm']
        available_cols = [col for col in cols_to_plot if col in df.columns]
        
        if 'timestep' not in df.columns:
            df['timestep'] = range(len(df))
        
        # Speed
        if 'speed' in df.columns:
            axes[0, 0].plot(df['timestep'], df['speed'], 'b-', linewidth=2)
            axes[0, 0].set_title('Speed (km/h)')
        else:
            axes[0, 0].text(0.5, 0.5, 'Speed\nNot Available', ha='center', va='center')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throttle
        if 'throttle' in df.columns:
            axes[0, 1].plot(df['timestep'], df['throttle'], 'g-', linewidth=2)
            axes[0, 1].set_title('Throttle (%)')
        else:
            axes[0, 1].text(0.5, 0.5, 'Throttle\nNot Available', ha='center', va='center')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Brake
        if 'brake' in df.columns:
            axes[0, 2].plot(df['timestep'], df['brake'], 'r-', linewidth=2)
            axes[0, 2].set_title('Brake')
        else:
            axes[0, 2].text(0.5, 0.5, 'Brake\nNot Available', ha='center', va='center')
        axes[0, 2].set_xlabel('Timestep')
        axes[0, 2].grid(True, alpha=0.3)
        
        # ERS
        if 'ers' in df.columns:
            axes[1, 0].plot(df['timestep'], df['ers'], 'm-', linewidth=2)
            axes[1, 0].set_title('ERS')
        else:
            axes[1, 0].text(0.5, 0.5, 'ERS\nNot Available', ha='center', va='center')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gap Ahead
        if 'gap_ahead' in df.columns:
            axes[1, 1].plot(df['timestep'], df['gap_ahead'], 'c-', linewidth=2)
            axes[1, 1].set_title('Gap Ahead (s)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Gap Ahead\nNot Available', ha='center', va='center')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Position Norm
        if 'position_norm' in df.columns:
            axes[1, 2].plot(df['timestep'], df['position_norm'], 'orange', linewidth=2)
            axes[1, 2].set_title('Position (Normalized)')
        else:
            axes[1, 2].text(0.5, 0.5, 'Position Norm\nNot Available', ha='center', va='center')
        axes[1, 2].set_xlabel('Timestep')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print sequence info
        print(f"\n=== SEQUENCE INFO ===")
        sequence = data[sequence_idx]
        for key in ['original_sequence_id', 'year', 'driver', 'original_length']:
            if key in sequence:
                print(f"{key}: {sequence[key]}")
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
    
    def prepare_sequences_for_ml(self, sequences: List[Dict], 
                               sequence_length: int = 50,
                               features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for machine learning (transformer input)
        
        Args:
            sequences: List of telemetry sequences
            sequence_length: Fixed length for sequences (pad/truncate)
            features: List of feature columns to use
            
        Returns:
            Tuple of (X, metadata) where X is feature array and metadata contains sequence info
        """
        if features is None:
            features = ['speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm']
        
        X_sequences = []
        metadata = []
        failed_count = 0
        
        for seq in sequences:
            df = self.sequence_to_dataframe(seq)
            
            if df.empty:
                failed_count += 1
                continue
            
            # Check which features are available
            available_features = [f for f in features if f in df.columns]
            if not available_features:
                failed_count += 1
                continue
            
            # Extract features
            feature_data = df[available_features].values
            
            # Pad missing features with zeros
            if len(available_features) < len(features):
                missing_features = len(features) - len(available_features)
                padding_features = np.zeros((len(feature_data), missing_features))
                feature_data = np.hstack([feature_data, padding_features])
            
            # Pad or truncate to fixed length
            if len(feature_data) > sequence_length:
                # Truncate
                feature_data = feature_data[:sequence_length]
            elif len(feature_data) < sequence_length:
                # Pad with zeros
                padding = np.zeros((sequence_length - len(feature_data), len(features)))
                feature_data = np.vstack([feature_data, padding])
            
            X_sequences.append(feature_data)
            
            # Store metadata
            seq_metadata = {
                'original_length': len(df),
                'driver': seq.get('driver', -1),
                'year': seq.get('year', -1),
                'sequence_id': seq.get('original_sequence_id', 'unknown'),
                'available_features': available_features
            }
            metadata.append(seq_metadata)
        
        if failed_count > 0:
            print(f"Warning: {failed_count} sequences failed during ML preparation")
        
        X = np.array(X_sequences)
        return X, metadata
    
    def save_processed_data(self, output_folder: str = "processed_data"):
        """Save processed data for ML training"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Prepare data for ML
        features = ['speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm']
        
        if self.train_data:
            X_train, meta_train = self.prepare_sequences_for_ml(self.train_data, features=features)
            if X_train.size > 0:
                np.save(output_path / "X_train.npy", X_train)
                with open(output_path / "meta_train.json", 'w') as f:
                    json.dump(meta_train, f, indent=2)
                print(f"Saved training data: {X_train.shape}")
        
        if self.test_data:
            X_test, meta_test = self.prepare_sequences_for_ml(self.test_data, features=features)
            if X_test.size > 0:
                np.save(output_path / "X_test.npy", X_test)
                with open(output_path / "meta_test.json", 'w') as f:
                    json.dump(meta_test, f, indent=2)
                print(f"Saved test data: {X_test.shape}")
        
        if self.validation_data:
            X_val, meta_val = self.prepare_sequences_for_ml(self.validation_data, features=features)
            if X_val.size > 0:
                np.save(output_path / "X_val.npy", X_val)
                with open(output_path / "meta_val.json", 'w') as f:
                    json.dump(meta_val, f, indent=2)
                print(f"Saved validation data: {X_val.shape}")
        
        print(f"Processed data saved to {output_path}")

# Usage Example
if __name__ == "__main__":
    # Initialize processor
    base_folder = "/Users/sid/Downloads/F1-DRS-Decision-AI/transformer_austria_training"
    processor = F1TelemetryProcessor(base_folder)
    
    # Debug data structure first
    print("🔍 Debugging data structure...")
    processor.debug_data_structure(max_files=2)
    
    # Load all data
    print("\n📁 Loading all data...")
    processor.load_all_data()
    
    # Try to analyze data distribution
    try:
        print("\n📊 Analyzing data distribution...")
        all_data_df = processor.analyze_data_distribution()
        
        # Visualize a sample sequence if we have data
        if processor.train_data and len(processor.train_data) > 0:
            print("\n📈 Creating sample visualization...")
            processor.visualize_sample_sequence(sequence_idx=0, split="train")
        
        # Prepare data for machine learning
        if processor.train_data:
            print("\n🤖 Preparing data for ML...")
            X_train, metadata_train = processor.prepare_sequences_for_ml(processor.train_data)
            if X_train.size > 0:
                print(f"Prepared training data shape: {X_train.shape}")
                print(f"Features per timestep: {X_train.shape[2]}")
                print(f"Sequence length: {X_train.shape[1]}")
                print(f"Number of sequences: {X_train.shape[0]}")
        
        # Save processed data
        print("\n💾 Saving processed data...")
        processor.save_processed_data()
        
        print("\n✅ Processing Complete!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("\nThis might be due to inconsistent data formats.")
        print("Check the debug output above to understand your data structure.")
        import traceback
        traceback.print_exc()