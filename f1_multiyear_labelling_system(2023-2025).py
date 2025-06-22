"""
Created on Sat Jun 21 14:14:45 2025

@author: sid

F1 Complete Multi-Year Labeling System (2023-2025) - Self-Contained
Complete integrated system for Austrian GP DRS prediction using all three years

Features:
- Self-contained (no external imports needed)
- Works with your actual file paths
- Unified labeling for 2023, 2024, and 2025 data
- Austrian GP prediction optimization
- Cross-year transfer learning
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_PATHS = {
    2023: {
        'features': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2023/2023_features.csv',
        'metadata': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2023/2023_metadata.csv',
        'output': '/Users/sid/Downloads/F1-DRS-Decision-AI/labeling_data_2023'
    },
    2024: {
        'features': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2024/2024_features_enhanced.csv',
        'metadata': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2024/2024_metadata_enhanced.csv',
        'output': '/Users/sid/Downloads/F1-DRS-Decision-AI/labeling_data_2024'
    },
    2025: {
        'features': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2025/2025_austria_features.csv',
        'metadata': '/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2025/2025_austria_metadata.csv',
        'output': '/Users/sid/Downloads/F1-DRS-Decision-AI/labeling_data_2025'
    }
}

INTEGRATED_OUTPUT_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/integrated_multi_year_labeling"
FINAL_TRAINING_OUTPUT = "/Users/sid/Downloads/F1-DRS-Decision-AI/final_austria_gp_training_data"

# Circuit similarity to Austria across all years
AUSTRIA_SIMILARITY_SCORES = {
    # 2023 circuits
    'austria': 1.0,
    'canada': 0.85,
    'italy': 0.8,
    'belgium': 0.75,
    'bahrain': 0.7,
    'saudi_arabia': 0.65,
    'australia': 0.6,
    'great_britain': 0.55,
    'netherlands': 0.5,
    'spain': 0.45,
    'hungary': 0.4,
    'monaco': 0.2,
    
    # 2024 additions
    'china': 0.85,  # High similarity to Austria
    'brazil': 0.6,
    'mexico': 0.45,
    'united_states': 0.4,
    'azerbaijan': 0.4,
    'singapore': 0.3,
    'japan': 0.3,
    'las_vegas': 0.25,
    'qatar': 0.25,
    'miami': 0.4
}

class F1CompleteMultiYearLabelingSystem:
    """Complete self-contained system for F1 multi-year labeling"""
    
    def __init__(self):
        self.output_path = Path(INTEGRATED_OUTPUT_PATH)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.final_output_path = Path(FINAL_TRAINING_OUTPUT)
        self.final_output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.output_path / "complete_multi_year_labeling.db"
        self.init_database()
        
        # Load all available datasets
        self.datasets = {}
        self.load_all_datasets()
        
        print("🏎️  F1 COMPLETE MULTI-YEAR LABELING SYSTEM")
        print("=" * 60)
        print("🎯 Mission: Austrian GP 2025 DRS Prediction")
        print("📊 Years: 2023 (Gold Standard) + 2024 (Enhanced) + 2025 (Target)")
        print("=" * 60)
        
    def init_database(self):
        """Initialize comprehensive database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Multi-year labeling table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_year_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT,
                year INTEGER,
                circuit TEXT,
                max_speed REAL,
                speed_range REAL,
                austria_similarity REAL,
                overtake_decision INTEGER,
                overtake_success INTEGER,
                label_confidence REAL,
                label_source TEXT,
                selection_priority REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Austria evolution tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS austria_evolution (
                year INTEGER PRIMARY KEY,
                sequence_count INTEGER,
                avg_max_speed REAL,
                avg_speed_range REAL,
                drs_usage_rate REAL,
                avg_confidence REAL,
                prediction_weight REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def load_all_datasets(self):
        """Load all available datasets"""
        
        for year, paths in DATA_PATHS.items():
            try:
                print(f"📊 Loading {year} data...")
                
                features_df = pd.read_csv(paths['features'])
                
                # Try to load metadata, but don't fail if it doesn't exist
                try:
                    metadata_df = pd.read_csv(paths['metadata'])
                except FileNotFoundError:
                    print(f"   ⚠️  Metadata not found for {year}, using features only")
                    metadata_df = None
                
                # Analyze the data
                austria_count = len(features_df[features_df['circuit'].str.lower() == 'austria'])
                
                self.datasets[year] = {
                    'features': features_df,
                    'metadata': metadata_df,
                    'total_sequences': len(features_df),
                    'circuits': features_df['circuit'].nunique(),
                    'austria_sequences': austria_count,
                    'unique_circuits': features_df['circuit'].str.lower().unique().tolist()
                }
                
                print(f"   ✅ {year}: {len(features_df):,} sequences, {features_df['circuit'].nunique()} circuits")
                if austria_count > 0:
                    print(f"      🇦🇹 Austria: {austria_count} sequences")
                print(f"      🏁 Circuits: {', '.join(sorted(features_df['circuit'].str.lower().unique()[:5]))}...")
                
            except FileNotFoundError as e:
                print(f"   ❌ {year} data not found: {e}")
                self.datasets[year] = None
        
        print(f"\n📈 Data Summary:")
        total_sequences = sum(data['total_sequences'] for data in self.datasets.values() if data)
        total_austria = sum(data['austria_sequences'] for data in self.datasets.values() if data)
        print(f"   Total sequences across all years: {total_sequences:,}")
        print(f"   Total Austria sequences: {total_austria:,}")

class MultiYearSequenceSelector:
    """Strategic sequence selection across all years for Austrian GP training"""
    
    def __init__(self, datasets: Dict):
        self.datasets = datasets
        
    def select_sequences_for_austria_training(self, total_target: int = 4000) -> pd.DataFrame:
        """
        Select optimal sequences across all years for Austrian GP training
        
        Strategy:
        - Prioritize ALL Austria sequences from any year
        - Focus on high-similarity circuits
        - Ensure representation from each year
        - Balance for optimal training diversity
        """
        
        print(f"🎯 Strategic Multi-Year Selection for Austrian GP Training")
        print(f"Target: {total_target:,} sequences across all years")
        
        all_selected = []
        selection_stats = {}
        
        # Year allocation based on data quality and relevance
        year_allocations = {
            2023: 0.25,  # Austria gold standard
            2024: 0.45,  # Large dataset with China GP
            2025: 0.30   # Direct target year
        }
        
        for year, allocation in year_allocations.items():
            if year not in self.datasets or not self.datasets[year]:
                print(f"   ⚠️  {year} data not available")
                continue
                
            year_target = int(total_target * allocation)
            features_df = self.datasets[year]['features']
            
            print(f"\n📊 {year} Selection (Target: {year_target:,})")
            
            # Add Austria similarity scores
            features_df = features_df.copy()
            features_df['austria_similarity'] = features_df['circuit'].str.lower().map(
                AUSTRIA_SIMILARITY_SCORES
            ).fillna(0.3)
            
            year_selected = self.select_year_sequences(features_df, year, year_target)
            
            if len(year_selected) > 0:
                year_selected['source_year'] = year
                all_selected.append(year_selected)
                
                # Statistics
                austria_count = len(year_selected[year_selected['circuit'].str.lower() == 'austria'])
                high_sim_count = len(year_selected[year_selected['austria_similarity'] > 0.7])
                
                selection_stats[year] = {
                    'total': len(year_selected),
                    'austria': austria_count,
                    'high_similarity': high_sim_count,
                    'circuits': year_selected['circuit'].nunique()
                }
                
                print(f"   ✅ Selected: {len(year_selected):,} sequences")
                print(f"      🇦🇹 Austria: {austria_count}")
                print(f"      🎯 High similarity: {high_sim_count}")
        
        # Combine all selections
        if not all_selected:
            print("❌ No sequences selected")
            return pd.DataFrame()
        
        final_selection = pd.concat(all_selected, ignore_index=True)
        
        # Final analysis
        print(f"\n📊 FINAL MULTI-YEAR SELECTION:")
        print(f"   Total selected: {len(final_selection):,}")
        
        total_austria = len(final_selection[final_selection['circuit'].str.lower() == 'austria'])
        total_high_sim = len(final_selection[final_selection['austria_similarity'] > 0.7])
        
        print(f"   🇦🇹 Total Austria: {total_austria}")
        print(f"   🎯 High similarity: {total_high_sim}")
        print(f"   📈 Year distribution:")
        
        for year, stats in selection_stats.items():
            print(f"      {year}: {stats['total']:,} sequences")
        
        # Circuit distribution across all years
        print(f"   🏁 Top circuits selected:")
        circuit_dist = final_selection['circuit'].str.lower().value_counts().head(8)
        for circuit, count in circuit_dist.items():
            similarity = AUSTRIA_SIMILARITY_SCORES.get(circuit, 0.3)
            emoji = "🥇" if circuit == 'austria' else "🎯" if similarity > 0.7 else "✅"
            print(f"      {emoji} {circuit}: {count} (sim: {similarity:.2f})")
        
        return final_selection
    
    def select_year_sequences(self, features_df: pd.DataFrame, year: int, target: int) -> pd.DataFrame:
        """Select sequences for a specific year"""
        
        selected_sequences = []
        selection_reasons = []
        
        # 1. ALL AUSTRIA SEQUENCES (Highest Priority)
        austria_df = features_df[features_df['circuit'].str.lower() == 'austria']
        if len(austria_df) > 0:
            selected_sequences.extend(austria_df.index.tolist())
            selection_reasons.extend(['austria_gold_standard'] * len(austria_df))
            print(f"      🥇 Austria sequences: {len(austria_df)} (ALL selected)")
        
        remaining_target = max(0, target - len(austria_df))
        
        if remaining_target > 0:
            # 2. High similarity circuits (60% of remaining)
            high_sim_target = int(remaining_target * 0.6)
            remaining_df = features_df[~features_df.index.isin(selected_sequences)]
            
            high_sim_circuits = ['canada', 'china', 'italy', 'belgium', 'bahrain']
            high_sim_df = remaining_df[
                remaining_df['circuit'].str.lower().isin(high_sim_circuits)
            ]
            
            if len(high_sim_df) > 0:
                # Priority based on Austria similarity and speed performance
                high_sim_df = high_sim_df.copy()
                high_sim_df['priority'] = (
                    high_sim_df['austria_similarity'] * 0.5 +
                    (high_sim_df['max_speed'] / 350.0) * 0.3 +
                    (high_sim_df['speed_range'] / 150.0) * 0.2
                )
                
                n_sample = min(high_sim_target, len(high_sim_df))
                sampled = high_sim_df.nlargest(n_sample, 'priority')
                selected_sequences.extend(sampled.index.tolist())
                selection_reasons.extend(['high_similarity'] * n_sample)
                
                print(f"      🎯 High similarity: {n_sample}")
            
            # 3. Diverse performance ranges (25% of remaining)
            diverse_target = int(remaining_target * 0.25)
            remaining_df = features_df[~features_df.index.isin(selected_sequences)]
            
            if len(remaining_df) > diverse_target:
                # Speed-based diversity
                try:
                    speed_bins = pd.qcut(remaining_df['max_speed'], q=4, duplicates='drop')
                    diverse_samples = remaining_df.groupby(speed_bins, observed=True).apply(
                        lambda x: x.sample(min(len(x), max(1, diverse_target // 4)))
                    ).reset_index(drop=True)
                    
                    n_diverse = min(diverse_target, len(diverse_samples))
                    selected_sequences.extend(diverse_samples.index[:n_diverse].tolist())
                    selection_reasons.extend(['diverse_performance'] * n_diverse)
                    
                    print(f"      📊 Diverse performance: {n_diverse}")
                except (ValueError, KeyError):
                    # Fallback to random sampling
                    n_sample = min(diverse_target, len(remaining_df))
                    sampled = remaining_df.sample(n_sample)
                    selected_sequences.extend(sampled.index.tolist())
                    selection_reasons.extend(['diverse_random'] * n_sample)
            
            # 4. Fill remaining with best available (15% of remaining)
            final_remaining = target - len(selected_sequences)
            if final_remaining > 0:
                remaining_df = features_df[~features_df.index.isin(selected_sequences)]
                if len(remaining_df) > 0:
                    n_sample = min(final_remaining, len(remaining_df))
                    
                    # Prefer higher quality sequences
                    if 'data_quality_score' in remaining_df.columns:
                        sampled = remaining_df.nlargest(n_sample, 'data_quality_score')
                    else:
                        sampled = remaining_df.sample(n_sample)
                    
                    selected_sequences.extend(sampled.index.tolist())
                    selection_reasons.extend(['fill_remaining'] * n_sample)
                    
                    print(f"      ⚡ Fill remaining: {n_sample}")
        
        # Create final selection
        final_selection = features_df.loc[selected_sequences].copy()
        final_selection['selection_reason'] = selection_reasons[:len(selected_sequences)]
        
        return final_selection

class MultiYearLabelGenerator:
    """Generate synthetic labels optimized for Austrian GP prediction across all years"""
    
    def __init__(self):
        self.year_patterns = self.define_year_patterns()
        
    def define_year_patterns(self) -> Dict:
        """Define year-specific patterns for labeling"""
        return {
            2023: {
                'base_speed_threshold': 270.0,
                'base_range_threshold': 40.0,
                'confidence_multiplier': 1.0,
                'characteristics': 'stable_regulations'
            },
            2024: {
                'base_speed_threshold': 275.0,
                'base_range_threshold': 42.0,
                'confidence_multiplier': 1.05,
                'characteristics': 'refined_aero'
            },
            2025: {
                'base_speed_threshold': 278.0,
                'base_range_threshold': 43.0,
                'confidence_multiplier': 1.1,
                'characteristics': 'optimized_performance'
            }
        }
    
    def generate_multi_year_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate labels for multi-year dataset"""
        
        print(f"🧬 Generating multi-year labels for {len(features_df):,} sequences...")
        
        labeled_df = features_df.copy()
        
        # Generate labels for each sequence
        labels = []
        for _, row in features_df.iterrows():
            label = self.generate_sequence_label(row)
            labels.append(label)
        
        # Add labels to dataframe
        labels_df = pd.DataFrame(labels)
        
        # Ensure we have the key columns
        for col in ['overtake_decision', 'overtake_success', 'label_confidence', 'label_source']:
            if col in labels_df.columns:
                labeled_df[col] = labels_df[col]
        
        # Statistics
        year_stats = {}
        if 'source_year' in labeled_df.columns:
            for year in labeled_df['source_year'].unique():
                year_data = labeled_df[labeled_df['source_year'] == year]
                year_stats[year] = {
                    'sequences': len(year_data),
                    'drs_rate': (year_data['overtake_decision'] >= 1).mean(),
                    'avg_confidence': year_data['label_confidence'].mean()
                }
        
        print(f"   ✅ Labels generated successfully")
        print(f"   📊 Overall DRS usage rate: {(labeled_df['overtake_decision'] >= 1).mean():.1%}")
        print(f"   📈 Average confidence: {labeled_df['label_confidence'].mean():.3f}")
        
        if year_stats:
            print(f"   📋 Year breakdown:")
            for year, stats in year_stats.items():
                print(f"      {year}: {stats['sequences']:,} seq, DRS: {stats['drs_rate']:.1%}, conf: {stats['avg_confidence']:.2f}")
        
        return labeled_df
    
    def generate_sequence_label(self, row: pd.Series) -> Dict:
        """Generate label for individual sequence"""
        
        circuit = row['circuit'].lower() if isinstance(row['circuit'], str) else 'unknown'
        max_speed = row.get('max_speed', 0)
        speed_range = row.get('speed_range', 0)
        year = row.get('source_year', 2024)
        austria_similarity = row.get('austria_similarity', AUSTRIA_SIMILARITY_SCORES.get(circuit, 0.3))
        
        # Get year-specific patterns
        year_pattern = self.year_patterns.get(year, self.year_patterns[2024])
        
        # Base thresholds adjusted by year
        speed_threshold = year_pattern['base_speed_threshold']
        range_threshold = year_pattern['base_range_threshold']
        confidence_mult = year_pattern['confidence_multiplier']
        
        # Circuit-specific adjustments
        if circuit == 'austria':
            # Gold standard - highest confidence
            confidence_base = 0.95
            speed_threshold *= 0.95  # Slightly lower threshold for Austria
        elif austria_similarity > 0.8:
            # Very high similarity (China, Canada)
            confidence_base = 0.90
            speed_threshold *= 0.98
        elif austria_similarity > 0.6:
            # High similarity
            confidence_base = 0.85
        elif austria_similarity > 0.4:
            # Medium similarity
            confidence_base = 0.75
            speed_threshold *= 1.05
        else:
            # Low similarity (Monaco, etc.)
            confidence_base = 0.65
            speed_threshold *= 1.1
        
        # Primary decision logic
        if max_speed > speed_threshold and speed_range > range_threshold:
            decision = 1
            confidence = confidence_base * 1.0
        elif max_speed > speed_threshold - 15 and speed_range > range_threshold - 10:
            decision = 1
            confidence = confidence_base * 0.85
        elif max_speed > speed_threshold + 20 and speed_range > range_threshold + 15:
            decision = 2 if row.get('zone_count', 1) > 1 else 1  # Multi-zone strategy
            confidence = confidence_base * 1.1
        elif max_speed < speed_threshold - 30:
            decision = 0
            confidence = confidence_base * 0.9
        else:
            # Default based on circuit characteristics
            if circuit in ['austria', 'canada', 'china', 'italy']:
                decision = 1
                confidence = confidence_base * 0.8
            elif circuit in ['monaco', 'singapore']:
                decision = 0
                confidence = confidence_base * 0.9
            else:
                decision = 1 if max_speed > 250 else 0
                confidence = confidence_base * 0.7
        
        # Success prediction
        if decision >= 1:
            success_factors = (
                (speed_range / 100.0) * 0.4 +
                austria_similarity * 0.4 +
                min(max_speed / 300.0, 1.0) * 0.2
            )
            
            if success_factors > 0.7:
                success = 1  # Successful
            elif success_factors > 0.4:
                success = 0  # Attempted but unsuccessful
            else:
                success = -1  # Unclear/not applicable
        else:
            success = -1  # No DRS usage
        
        # Apply year-specific confidence multiplier
        confidence *= confidence_mult
        confidence = min(confidence, 0.98)  # Cap at 98%
        
        return {
            'overtake_decision': decision,
            'overtake_success': success,
            'label_confidence': confidence,
            'label_source': f'synthetic_multi_year_{year}'
        }

class IntegratedTrainingDataExporter:
    """Export comprehensive training datasets for Austrian GP prediction"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def export_training_datasets(self, labeled_df: pd.DataFrame) -> Dict:
        """Export comprehensive training datasets"""
        
        print(f"📤 Exporting Austrian GP training datasets...")
        
        # Main dataset
        labeled_df.to_csv(self.output_path / "COMPLETE_MULTI_YEAR_DATASET.csv", index=False)
        
        # Specialized datasets
        datasets = self.create_specialized_datasets(labeled_df)
        
        # Export each specialized dataset
        for name, dataset in datasets.items():
            filename = f"{name}.csv"
            dataset.to_csv(self.output_path / filename, index=False)
        
        # Create comprehensive summary
        summary = self.create_export_summary(labeled_df, datasets)
        
        # Save summary
        with open(self.output_path / "training_datasets_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create training guide
        self.create_training_guide(summary)
        
        print(f"   ✅ Exported {len(datasets) + 1} datasets")
        print(f"   📊 Total sequences: {len(labeled_df):,}")
        
        return summary
    
    def create_specialized_datasets(self, labeled_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create specialized training datasets"""
        
        datasets = {}
        
        # 1. Austria Gold Standard (highest priority)
        austria_data = labeled_df[labeled_df['circuit'].str.lower() == 'austria'].copy()
        if len(austria_data) > 0:
            datasets['AUSTRIA_GOLD_STANDARD'] = austria_data
        
        # 2. High Austria Similarity (Canada, China, Italy)
        high_sim_data = labeled_df[labeled_df['austria_similarity'] > 0.7].copy()
        datasets['HIGH_AUSTRIA_SIMILARITY'] = high_sim_data
        
        # 3. High Confidence Multi-Year
        high_conf_data = labeled_df[labeled_df['label_confidence'] > 0.85].copy()
        datasets['HIGH_CONFIDENCE'] = high_conf_data
        
        # 4. Year-specific datasets
        if 'source_year' in labeled_df.columns:
            for year in labeled_df['source_year'].unique():
                year_data = labeled_df[labeled_df['source_year'] == year].copy()
                datasets[f'YEAR_{year}_SPECIFIC'] = year_data
        
        # 5. Circuit-type specific
        high_speed_circuits = ['austria', 'canada', 'china', 'italy', 'belgium']
        high_speed_data = labeled_df[
            labeled_df['circuit'].str.lower().isin(high_speed_circuits)
        ].copy()
        datasets['HIGH_SPEED_CIRCUITS'] = high_speed_data
        
        # 6. Balanced training set
        balanced_data = self.create_balanced_dataset(labeled_df)
        datasets['BALANCED_TRAINING'] = balanced_data
        
        # 7. Transfer learning ready
        transfer_ready = labeled_df[
            (labeled_df['label_confidence'] > 0.8) & 
            (labeled_df['austria_similarity'] > 0.6) &
            (labeled_df['overtake_decision'] != -1)
        ].copy()
        datasets['TRANSFER_LEARNING_READY'] = transfer_ready
        
        return datasets
    
    def create_balanced_dataset(self, labeled_df: pd.DataFrame, target_size: int = 2500) -> pd.DataFrame:
        """Create balanced dataset across years and similarity levels"""
        
        # Create similarity bins
        labeled_df_copy = labeled_df.copy()
        labeled_df_copy['similarity_bin'] = pd.cut(
            labeled_df_copy['austria_similarity'], 
            bins=[0, 0.4, 0.7, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        balanced_samples = []
        
        # Sample proportionally from each bin
        for sim_bin in ['high', 'medium', 'low']:
            bin_data = labeled_df_copy[labeled_df_copy['similarity_bin'] == sim_bin]
            if len(bin_data) > 0:
                # Allocate more samples to higher similarity
                if sim_bin == 'high':
                    n_sample = min(target_size // 2, len(bin_data))
                elif sim_bin == 'medium':
                    n_sample = min(target_size // 3, len(bin_data))
                else:
                    n_sample = min(target_size // 6, len(bin_data))
                
                if len(bin_data) > n_sample:
                    sampled = bin_data.sample(n_sample)
                else:
                    sampled = bin_data
                
                balanced_samples.append(sampled)
        
        if balanced_samples:
            balanced_dataset = pd.concat(balanced_samples, ignore_index=True)
        else:
            balanced_dataset = labeled_df.sample(min(target_size, len(labeled_df)))
        
        return balanced_dataset
    
    def create_export_summary(self, main_dataset: pd.DataFrame, specialized_datasets: Dict) -> Dict:
        """Create comprehensive export summary"""
        
        summary = {
            'export_date': datetime.now().isoformat(),
            'main_dataset': {
                'filename': 'COMPLETE_MULTI_YEAR_DATASET.csv',
                'total_sequences': len(main_dataset),
                'labeled_sequences': (main_dataset['overtake_decision'] != -1).sum(),
                'avg_confidence': main_dataset['label_confidence'].mean(),
                'drs_usage_rate': (main_dataset['overtake_decision'] >= 1).mean(),
                'circuits': main_dataset['circuit'].nunique(),
                'austria_sequences': len(main_dataset[main_dataset['circuit'].str.lower() == 'austria'])
            }
        }
        
        # Year distribution
        if 'source_year' in main_dataset.columns:
            year_dist = main_dataset['source_year'].value_counts().to_dict()
            summary['year_distribution'] = year_dist
        
        # Circuit distribution
        circuit_dist = main_dataset['circuit'].str.lower().value_counts().head(10).to_dict()
        summary['top_circuits'] = circuit_dist
        
        # Specialized datasets summary
        summary['specialized_datasets'] = {}
        for name, dataset in specialized_datasets.items():
            summary['specialized_datasets'][name] = {
                'filename': f'{name}.csv',
                'sequences': len(dataset),
                'austria_sequences': len(dataset[dataset['circuit'].str.lower() == 'austria']),
                'avg_confidence': dataset['label_confidence'].mean(),
                'drs_usage_rate': (dataset['overtake_decision'] >= 1).mean()
            }
        
        # Quality metrics
        summary['quality_metrics'] = {
            'high_confidence_sequences': (main_dataset['label_confidence'] > 0.85).sum(),
            'austria_similarity_high': (main_dataset['austria_similarity'] > 0.7).sum(),
            'multi_year_coverage': main_dataset['source_year'].nunique() if 'source_year' in main_dataset.columns else 1,
            'readiness_score': self.calculate_readiness_score(main_dataset)
        }
        
        return summary
    
    def calculate_readiness_score(self, dataset: pd.DataFrame) -> float:
        """Calculate Austrian GP prediction readiness score"""
        
        score = 0.0
        
        # Austria sequences (30%)
        austria_count = len(dataset[dataset['circuit'].str.lower() == 'austria'])
        austria_factor = min(austria_count / 50, 1.0) * 0.3
        score += austria_factor
        
        # High similarity coverage (25%)
        high_sim_count = len(dataset[dataset['austria_similarity'] > 0.7])
        similarity_factor = min(high_sim_count / 800, 1.0) * 0.25
        score += similarity_factor
        
        # Label quality (25%)
        avg_confidence = dataset['label_confidence'].mean()
        confidence_factor = avg_confidence * 0.25
        score += confidence_factor
        
        # Dataset size (20%)
        size_factor = min(len(dataset) / 3000, 1.0) * 0.2
        score += size_factor
        
        return score
    
    def create_training_guide(self, summary: Dict):
        """Create training guide for users"""
        
        guide_content = f"""# F1 Austrian GP Training Datasets Guide

## 📊 Dataset Overview
- **Total Sequences**: {summary['main_dataset']['total_sequences']:,}
- **Austria Sequences**: {summary['main_dataset']['austria_sequences']:,}
- **Average Confidence**: {summary['main_dataset']['avg_confidence']:.3f}
- **DRS Usage Rate**: {summary['main_dataset']['drs_usage_rate']:.1%}
- **Readiness Score**: {summary['quality_metrics']['readiness_score']:.2f}/1.0

## 🎯 Recommended Training Workflow

### 1. Start with Gold Standard
**File**: `AUSTRIA_GOLD_STANDARD.csv`
- Direct Austria GP data across all years
- Highest relevance for 2025 prediction
- Use for initial model validation

### 2. Primary Training
**File**: `HIGH_AUSTRIA_SIMILARITY.csv`
- Canada, China, Italy circuits (similarity > 0.7)
- Best transfer learning candidates
- Core training dataset

### 3. Enhanced Training
**File**: `HIGH_CONFIDENCE.csv`
- High-quality labels (confidence > 0.85)
- Reduced noise, better convergence
- For model refinement

### 4. Full Dataset Training
**File**: `COMPLETE_MULTI_YEAR_DATASET.csv`
- Complete multi-year coverage
- Maximum data diversity
- For final model training

## 📋 Dataset Descriptions

"""
        
        for name, info in summary['specialized_datasets'].items():
            guide_content += f"""
### {name}
- **Sequences**: {info['sequences']:,}
- **Austria Data**: {info['austria_sequences']:,}
- **Confidence**: {info['avg_confidence']:.3f}
- **DRS Rate**: {info['drs_usage_rate']:.1%}
"""
        
        guide_content += """

## 🚀 ML Training Tips

1. **Start Small**: Begin with AUSTRIA_GOLD_STANDARD for proof of concept
2. **Progressive Training**: Gradually include HIGH_AUSTRIA_SIMILARITY, then BALANCED_TRAINING
3. **Validation Strategy**: Use year-based splits (e.g., train on 2023-2024, validate on 2025)
4. **Feature Engineering**: Focus on speed_range, max_speed, and austria_similarity features
5. **Transfer Learning**: Use pre-trained models from similar circuits

## 🎯 Austrian GP 2025 Prediction Focus

The datasets are optimized for predicting DRS decisions at the 2025 Austrian Grand Prix:
- **High Priority**: Austria, Canada, China sequences
- **Medium Priority**: Italy, Belgium, Bahrain sequences  
- **Context**: Monaco, Singapore for contrast learning

Good luck with your Austrian GP DRS prediction model! 🏎️
"""
        
        with open(self.output_path / "TRAINING_GUIDE.md", 'w') as f:
            f.write(guide_content)

def main_complete_workflow():
    """Run the complete multi-year labeling workflow"""
    
    print("🏎️  F1 COMPLETE MULTI-YEAR LABELING WORKFLOW")
    print("🎯 Austrian GP 2025 DRS Prediction Training Data Generation")
    print("=" * 70)
    
    # Initialize system
    system = F1CompleteMultiYearLabelingSystem()
    
    # Check if we have any data
    available_years = [year for year, data in system.datasets.items() if data is not None]
    if not available_years:
        print("❌ No data available. Please check your file paths.")
        return
    
    print(f"✅ Available years: {available_years}")
    
    # Strategic sequence selection
    print(f"\n🎯 PHASE 1: Strategic Sequence Selection")
    selector = MultiYearSequenceSelector(system.datasets)
    selected_sequences = selector.select_sequences_for_austria_training()
    
    if len(selected_sequences) == 0:
        print("❌ No sequences selected")
        return
    
    # Generate labels
    print(f"\n🧬 PHASE 2: Multi-Year Label Generation")
    labeler = MultiYearLabelGenerator()
    labeled_dataset = labeler.generate_multi_year_labels(selected_sequences)
    
    # Export training datasets
    print(f"\n📤 PHASE 3: Export Training Datasets")
    exporter = IntegratedTrainingDataExporter(system.final_output_path)
    export_summary = exporter.export_training_datasets(labeled_dataset)
    
    # Save to database
    print(f"\n💾 PHASE 4: Database Storage")
    conn = sqlite3.connect(system.db_path)
    labeled_dataset.to_sql('multi_year_labels', conn, if_exists='replace', index=False)
    conn.close()
    
    # Final summary
    print(f"\n🏁 COMPLETE WORKFLOW FINISHED!")
    print("=" * 70)
    print(f"📊 **FINAL RESULTS**:")
    print(f"   🎯 Total sequences: {len(labeled_dataset):,}")
    print(f"   🇦🇹 Austria sequences: {export_summary['main_dataset']['austria_sequences']:,}")
    print(f"   📈 Average confidence: {export_summary['main_dataset']['avg_confidence']:.3f}")
    print(f"   🏎️  DRS usage rate: {export_summary['main_dataset']['drs_usage_rate']:.1%}")
    print(f"   ⭐ Readiness score: {export_summary['quality_metrics']['readiness_score']:.2f}/1.0")
    
    print(f"\n📁 **TRAINING DATASETS CREATED**:")
    for name, info in export_summary['specialized_datasets'].items():
        print(f"   • {name}: {info['sequences']:,} sequences")
    
    print(f"\n🚀 **READY FOR MACHINE LEARNING!**")
    print(f"📂 Training data location: {system.final_output_path}")
    print(f"📖 See TRAINING_GUIDE.md for detailed instructions")
    print(f"🎯 Focus: Austrian GP 2025 DRS Decision Prediction")
    
    return {
        'labeled_dataset': labeled_dataset,
        'export_summary': export_summary,
        'output_path': system.final_output_path
    }

if __name__ == "__main__":
    # Run the complete workflow
    results = main_complete_workflow()