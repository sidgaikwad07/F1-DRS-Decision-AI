"""
Created on Mon Jun 23 09:39:31 2025

@author: sid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from f1_telemetry_processor import F1TelemetryProcessor

class DRSFeatureEngineer:
    """
    Feature engineering specifically for DRS decision making
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def engineer_drs_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features relevant for DRS decisions
        
        Args:
            df: DataFrame with telemetry data
            
        Returns:
            DataFrame with additional engineered features
        """
        df_enhanced = df.copy()
        
        # Speed-based features
        df_enhanced['speed_change'] = df_enhanced['speed'].diff().fillna(0)
        df_enhanced['speed_acceleration'] = df_enhanced['speed_change'].diff().fillna(0)
        df_enhanced['is_high_speed'] = (df_enhanced['speed'] > 200).astype(int)
        
        # Throttle-based features  
        df_enhanced['throttle_change'] = df_enhanced['throttle'].diff().fillna(0)
        df_enhanced['is_full_throttle'] = (df_enhanced['throttle'] > 90).astype(int)
        df_enhanced['throttle_stability'] = df_enhanced['throttle'].rolling(window=3).std().fillna(0)
        
        # Brake-based features
        df_enhanced['is_braking'] = (df_enhanced['brake'] > 0.1).astype(int)
        df_enhanced['brake_intensity'] = df_enhanced['brake']
        
        # Gap-based features (critical for DRS)
        df_enhanced['gap_change'] = df_enhanced['gap_ahead'].diff().fillna(0)
        df_enhanced['is_within_drs_zone'] = (df_enhanced['gap_ahead'] <= 1.0).astype(int)
        df_enhanced['gap_closing_rate'] = -df_enhanced['gap_change']  # Negative gap_change means closing
        
        # Position-based features
        df_enhanced['position_change'] = df_enhanced['position_norm'].diff().fillna(0)
        df_enhanced['is_gaining_position'] = (df_enhanced['position_change'] > 0).astype(int)
        
        # ERS-based features
        df_enhanced['ers_available'] = (df_enhanced['ers'] > 0).astype(int)
        
        # Combined features for DRS decision
        df_enhanced['drs_opportunity_score'] = (
            df_enhanced['is_within_drs_zone'] * 0.4 +
            df_enhanced['is_high_speed'] * 0.3 +
            df_enhanced['is_full_throttle'] * 0.2 +
            (1 - df_enhanced['is_braking']) * 0.1
        )
        
        # Rolling averages for stability
        window = 3
        for col in ['speed', 'throttle', 'gap_ahead']:
            df_enhanced[f'{col}_rolling_mean'] = df_enhanced[col].rolling(window=window).mean().fillna(df_enhanced[col])
        
        return df_enhanced
    
    def fit_scaler(self, df: pd.DataFrame, feature_columns: List[str]):
        """Fit the scaler on training data"""
        self.scaler.fit(df[feature_columns])
        self.fitted = True
        
    def transform_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(df[feature_columns])

class DRSLabelGenerator:
    """
    Generate DRS decision labels based on telemetry data
    """
    
    @staticmethod
    def generate_drs_labels(df: pd.DataFrame, 
                           gap_threshold: float = 1.0,
                           speed_threshold: float = 200,
                           brake_threshold: float = 0.1) -> pd.Series:
        """
        Generate DRS usage labels based on racing conditions
        
        Args:
            df: DataFrame with telemetry data
            gap_threshold: Maximum gap to car ahead for DRS eligibility (seconds)
            speed_threshold: Minimum speed for DRS usage (km/h)
            brake_threshold: Maximum brake input for DRS usage
            
        Returns:
            Series with binary DRS labels (1 = should use DRS, 0 = should not)
        """
        # Basic DRS conditions
        within_gap = df['gap_ahead'] <= gap_threshold
        sufficient_speed = df['speed'] >= speed_threshold
        not_braking = df['brake'] <= brake_threshold
        full_throttle = df['throttle'] > 80  # Mostly full throttle
        
        # Additional strategic conditions
        straight_line = df['speed_change'].abs() < 5  # Relatively stable speed (straight)
        
        # Combine conditions
        drs_should_be_used = (
            within_gap & 
            sufficient_speed & 
            not_braking & 
            full_throttle & 
            straight_line
        ).astype(int)
        
        return drs_should_be_used

def create_sequence_windows(data: np.ndarray, 
                          labels: np.ndarray, 
                          window_size: int = 10,
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for sequence prediction
    
    Args:
        data: Input feature array (samples, timesteps, features)
        labels: Target labels (samples, timesteps)
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        Tuple of (windowed_data, windowed_labels)
    """
    windowed_data = []
    windowed_labels = []
    
    for i in range(len(data)):
        sequence = data[i]
        sequence_labels = labels[i]
        
        for start in range(0, len(sequence) - window_size + 1, stride):
            end = start + window_size
            windowed_data.append(sequence[start:end])
            windowed_labels.append(sequence_labels[start:end])
    
    return np.array(windowed_data), np.array(windowed_labels)

def analyze_drs_opportunities(processor, split: str = "train"):
    """
    Analyze DRS opportunities in the dataset
    
    Args:
        processor: F1TelemetryProcessor instance
        split: Which data split to analyze
    """
    if split == "train" and processor.train_data:
        data = processor.train_data
    elif split == "test" and processor.test_data:
        data = processor.test_data
    elif split == "validation" and processor.validation_data:
        data = processor.validation_data
    else:
        print(f"No {split} data available")
        return
    
    feature_engineer = DRSFeatureEngineer()
    label_generator = DRSLabelGenerator()
    
    all_opportunities = []
    all_gaps = []
    all_speeds = []
    
    for sequence in data:
        df = processor.sequence_to_dataframe(sequence)
        
        # Engineer features
        df_enhanced = feature_engineer.engineer_drs_features(df)
        
        # Generate labels
        drs_labels = label_generator.generate_drs_labels(df_enhanced)
        
        # Collect statistics
        all_opportunities.extend(drs_labels.values)
        all_gaps.extend(df_enhanced['gap_ahead'].values)
        all_speeds.extend(df_enhanced['speed'].values)
    
    # Analysis
    total_timesteps = len(all_opportunities)
    drs_opportunities = sum(all_opportunities)
    drs_percentage = (drs_opportunities / total_timesteps) * 100
    
    print(f"\n=== DRS OPPORTUNITY ANALYSIS ({split}) ===")
    print(f"Total timesteps: {total_timesteps}")
    print(f"DRS opportunities: {drs_opportunities}")
    print(f"DRS opportunity percentage: {drs_percentage:.2f}%")
    print(f"Average gap ahead: {np.mean(all_gaps):.2f}s")
    print(f"Average speed: {np.mean(all_speeds):.1f} km/h")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # DRS opportunities distribution
    axes[0].hist(all_opportunities, bins=2, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('DRS Opportunities Distribution')
    axes[0].set_xlabel('DRS Should Be Used (0=No, 1=Yes)')
    axes[0].set_ylabel('Frequency')
    
    # Gap ahead distribution
    axes[1].hist(all_gaps, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=1.0, color='red', linestyle='--', label='DRS Threshold (1.0s)')
    axes[1].set_title('Gap Ahead Distribution')
    axes[1].set_xlabel('Gap to Car Ahead (seconds)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # Speed distribution
    axes[2].hist(all_speeds, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(x=200, color='red', linestyle='--', label='DRS Speed Threshold')
    axes[2].set_title('Speed Distribution')
    axes[2].set_xlabel('Speed (km/h)')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

# Example of a simple PyTorch transformer for DRS prediction
class DRSTransformer(nn.Module):
    """
    Simple transformer model for DRS decision prediction
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(DRSTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)  # (batch_size, seq_len)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)].transpose(0, 1)
        return self.dropout(x)

# Usage example for complete pipeline
def complete_drs_pipeline(base_folder: str):
    """
    Complete pipeline for DRS decision AI
    """
    print("=== F1 DRS DECISION AI PIPELINE ===")
    
    # 1. Load and process data
    processor = F1TelemetryProcessor(base_folder)
    processor.load_all_data()
    
    # 2. Analyze DRS opportunities
    analyze_drs_opportunities(processor, "train")
    
    # 3. Feature engineering and label generation
    feature_engineer = DRSFeatureEngineer()
    label_generator = DRSLabelGenerator()
    
    if processor.train_data:
        # Process training data
        train_sequences = []
        train_labels = []
        
        for sequence in processor.train_data:
            df = processor.sequence_to_dataframe(sequence)
            df_enhanced = feature_engineer.engineer_drs_features(df)
            
            # Fit scaler on first sequence
            feature_columns = ['speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm', 
                             'speed_change', 'throttle_change', 'gap_change', 'drs_opportunity_score']
            
            if len(train_sequences) == 0:
                feature_engineer.fit_scaler(df_enhanced, feature_columns)
            
            # Transform features
            features = feature_engineer.transform_features(df_enhanced, feature_columns)
            labels = label_generator.generate_drs_labels(df_enhanced)
            
            train_sequences.append(features)
            train_labels.append(labels.values)
        
        print(f"Processed {len(train_sequences)} training sequences")
        print(f"Feature dimension: {train_sequences[0].shape[1]}")
        
    return processor, feature_engineer, label_generator

if __name__ == "__main__":
    # Run complete pipeline
    base_folder = "/Users/sid/Downloads/F1-DRS-Decision-AI/transformer_austria_training"
    processor, engineer, generator = complete_drs_pipeline(base_folder)