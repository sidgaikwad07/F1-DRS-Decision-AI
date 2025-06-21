"""
Created on Sat Jun 21 11:35:49 2025

@author: sid
F1 2023 Robust Data Processor with Debugging
Handles various data structure variations and provides detailed debugging
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/F1_2023_3DRS_Dataset"
OUTPUT_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2023"
DEBUG_MODE = True  # Set to True to see detailed debugging info
SAMPLE_PROBLEMATIC_SEQUENCES = True  # Save examples of problematic sequences

def safe_convert_to_int(value: Any, default: int = 0, field_name: str = "unknown") -> int:
    """Safely convert any value to integer with debugging"""
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return int(value)
    
    if isinstance(value, str):
        try:
            return int(float(value))  # Handle string numbers
        except ValueError:
            if DEBUG_MODE:
                print(f"   ⚠️  {field_name}: Can't convert string '{value}' to int, using {default}")
            return default
    
    if isinstance(value, dict):
        if DEBUG_MODE:
            print(f"   🔍 {field_name}: Got dict {list(value.keys())[:3]}..., using {default}")
        # Maybe the value is nested in the dict
        if 'value' in value:
            return safe_convert_to_int(value['value'], default, f"{field_name}.value")
        elif len(value) == 1:
            # Single key dict, try the value
            key = list(value.keys())[0]
            return safe_convert_to_int(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list):
        if DEBUG_MODE:
            print(f"   🔍 {field_name}: Got list of length {len(value)}, using {default}")
        if len(value) > 0:
            return safe_convert_to_int(value[0], default, f"{field_name}[0]")
        return default
    
    if DEBUG_MODE:
        print(f"   ⚠️  {field_name}: Unknown type {type(value)}, using {default}")
    return default

def safe_convert_to_float(value: Any, default: float = 0.0, field_name: str = "unknown") -> float:
    """Safely convert any value to float"""
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    
    if isinstance(value, dict):
        if 'value' in value:
            return safe_convert_to_float(value['value'], default, f"{field_name}.value")
        elif len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_float(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_float(value[0], default, f"{field_name}[0]")
    
    return default

def safe_convert_to_string(value: Any, default: str = "unknown", field_name: str = "unknown") -> str:
    """Safely convert any value to string"""
    if value is None:
        return default
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, dict):
        if 'value' in value:
            return safe_convert_to_string(value['value'], default, f"{field_name}.value")
        elif 'name' in value:
            return safe_convert_to_string(value['name'], default, f"{field_name}.name")
        elif len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_string(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_string(value[0], default, f"{field_name}[0]")
    
    return str(value)

def extract_telemetry_data(data: Any, field_name: str = "telemetry") -> List[float]:
    """Extract telemetry data from various nested structures"""
    if not data:
        return []
    
    # Direct list
    if isinstance(data, list):
        try:
            return [float(x) for x in data if x is not None]
        except (ValueError, TypeError):
            return []
    
    # Nested in dict
    if isinstance(data, dict):
        # Common telemetry field names
        possible_keys = ['data', 'values', 'trace', 'series', field_name.split('_')[-1]]
        
        for key in possible_keys:
            if key in data:
                return extract_telemetry_data(data[key], f"{field_name}.{key}")
        
        # If no standard keys, try first list value
        for key, value in data.items():
            if isinstance(value, list):
                return extract_telemetry_data(value, f"{field_name}.{key}")
    
    return []

# ================================
# DATA CLASSES
# ================================

@dataclass
class DRSSequence:
    """Data class for F1 DRS sequence"""
    sequence_id: str
    circuit: str
    session: str
    lap_number: float
    driver: str
    year: int
    zone_index: int
    zone_count: int
    
    # Telemetry sequences
    speed_trace: List[float]
    throttle_trace: List[float] 
    brake_trace: List[float]
    ers_deployment: List[float]
    gap_to_ahead: List[float]
    
    # Context
    position: int
    drs_zones: List[int]
    
    # Labels
    label: int = -1
    sequence_length: int = 0
    data_quality_score: float = 0.0

class F1RobustProcessor:
    """Robust processor that handles various data formats"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.debug_info = {
            'problematic_sequences': [],
            'field_type_analysis': {},
            'processing_errors': []
        }
        
    def analyze_sequence_structure(self, sequence: Dict, sequence_index: int) -> Dict:
        """Analyze the structure of a sequence for debugging"""
        analysis = {
            'sequence_index': sequence_index,
            'top_level_keys': list(sequence.keys()),
            'field_types': {},
            'nested_structures': {}
        }
        
        for key, value in sequence.items():
            analysis['field_types'][key] = type(value).__name__
            
            if isinstance(value, dict):
                analysis['nested_structures'][key] = list(value.keys())[:5]  # First 5 keys
            elif isinstance(value, list) and len(value) > 0:
                analysis['nested_structures'][key] = f"List[{len(value)}] of {type(value[0]).__name__}"
        
        return analysis
    
    def load_data_safely(self) -> List[Dict]:
        """Load data with robust error handling"""
        all_sequences = []
        
        print(f"🔄 Loading data robustly from {self.data_path}...")
        
        # Load training sequences file
        training_file = self.data_path / "F1_2023_3DRS_training_sequences.json"
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    training_data = json.load(f)
                
                print(f"📁 Processing {training_file.name}...")
                
                if 'sequences' in training_data and isinstance(training_data['sequences'], list):
                    sequences = training_data['sequences']
                    print(f"   ✅ Found {len(sequences)} sequences in 'sequences' field")
                    
                    # Debug first few sequences
                    if DEBUG_MODE and len(sequences) > 0:
                        print(f"   🔍 Analyzing first sequence structure...")
                        first_seq_analysis = self.analyze_sequence_structure(sequences[0], 0)
                        print(f"      Keys: {first_seq_analysis['top_level_keys']}")
                        print(f"      Types: {first_seq_analysis['field_types']}")
                        if first_seq_analysis['nested_structures']:
                            print(f"      Nested: {first_seq_analysis['nested_structures']}")
                    
                    all_sequences.extend(sequences)
                
            except Exception as e:
                print(f"❌ Error loading {training_file.name}: {e}")
        
        # Load complete dataset file (if needed)
        complete_file = self.data_path / "F1_2023_3DRS_complete_dataset.json"
        if complete_file.exists() and len(all_sequences) == 0:
            print(f"📁 Trying {complete_file.name}...")
            try:
                with open(complete_file, 'r') as f:
                    complete_data = json.load(f)
                
                # This file has circuit-based structure, need to extract sequences
                for circuit_key, circuit_data in complete_data.items():
                    if isinstance(circuit_data, dict):
                        sequences = self._extract_sequences_from_circuit_data(circuit_data, circuit_key)
                        if sequences:
                            print(f"   ✅ {circuit_key}: {len(sequences)} sequences")
                            all_sequences.extend(sequences)
                
            except Exception as e:
                print(f"❌ Error loading {complete_file.name}: {e}")
        
        print(f"📊 Total sequences loaded: {len(all_sequences)}")
        return all_sequences
    
    def _extract_sequences_from_circuit_data(self, circuit_data: Dict, circuit_name: str) -> List[Dict]:
        """Extract sequences from circuit-based data structure"""
        sequences = []
        
        def search_for_sequences(obj, path=""):
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) and self._looks_like_sequence(item):
                        item['circuit'] = circuit_name.split('_')[0]  # Extract circuit name
                        sequences.append(item)
                    else:
                        search_for_sequences(item, f"{path}[{i}]")
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    search_for_sequences(value, f"{path}.{key}")
        
        search_for_sequences(circuit_data)
        return sequences
    
    def _looks_like_sequence(self, obj: Dict) -> bool:
        """Check if object looks like a sequence"""
        required_indicators = ['circuit', 'driver']
        telemetry_indicators = ['telemetry', 'speed', 'speed_trace']
        
        has_required = any(indicator in obj for indicator in required_indicators)
        has_telemetry = any(indicator in obj for indicator in telemetry_indicators)
        
        return has_required or has_telemetry
    
    def process_sequence_safely(self, raw_seq: Dict, index: int) -> Optional[DRSSequence]:
        """Process a single sequence with robust error handling"""
        
        if DEBUG_MODE and index < 3:  # Debug first 3 sequences
            print(f"\n🔍 Debug sequence {index}:")
            analysis = self.analyze_sequence_structure(raw_seq, index)
            print(f"   Keys: {analysis['top_level_keys']}")
            print(f"   Types: {analysis['field_types']}")
        
        try:
            # Extract basic metadata with safe conversion
            sequence_id = safe_convert_to_string(raw_seq.get('sequence_id', f"seq_{index}"), f"seq_{index}", "sequence_id")
            circuit = safe_convert_to_string(raw_seq.get('circuit', 'unknown'), 'unknown', "circuit")
            session = safe_convert_to_string(raw_seq.get('session', 'race'), 'race', "session")
            lap_number = safe_convert_to_float(raw_seq.get('lap_number', 0), 0.0, "lap_number")
            driver = safe_convert_to_string(raw_seq.get('driver', 'unknown'), 'unknown', "driver")
            year = safe_convert_to_int(raw_seq.get('year', 2023), 2023, "year")
            zone_index = safe_convert_to_int(raw_seq.get('zone_index', 0), 0, "zone_index")
            zone_count = safe_convert_to_int(raw_seq.get('zone_count', 0), 0, "zone_count")
            label = safe_convert_to_int(raw_seq.get('label', -1), -1, "label")
            position = safe_convert_to_int(raw_seq.get('position', 99), 99, "position")
            
            # Extract telemetry data from nested structure
            speed_trace = []
            throttle_trace = []
            brake_trace = []
            ers_deployment = []
            gap_to_ahead = []
            
            # Check for telemetry in nested 'telemetry' field
            if 'telemetry' in raw_seq:
                telemetry_data = raw_seq['telemetry']
                if isinstance(telemetry_data, dict):
                    speed_trace = extract_telemetry_data(telemetry_data.get('speed', []), "speed")
                    throttle_trace = extract_telemetry_data(telemetry_data.get('throttle', []), "throttle")
                    brake_trace = extract_telemetry_data(telemetry_data.get('brake', []), "brake")
                    ers_deployment = extract_telemetry_data(telemetry_data.get('ers', []), "ers")
                    gap_to_ahead = extract_telemetry_data(telemetry_data.get('gap_ahead', []), "gap_ahead")
            
            # Check for telemetry at top level as fallback
            if not speed_trace:
                speed_trace = extract_telemetry_data(raw_seq.get('speed_trace', []), "speed_trace")
            if not throttle_trace:
                throttle_trace = extract_telemetry_data(raw_seq.get('throttle_trace', []), "throttle_trace")
            if not brake_trace:
                brake_trace = extract_telemetry_data(raw_seq.get('brake_trace', []), "brake_trace")
            if not ers_deployment:
                ers_deployment = extract_telemetry_data(raw_seq.get('ers_deployment', []), "ers_deployment")
            if not gap_to_ahead:
                gap_to_ahead = extract_telemetry_data(raw_seq.get('gap_to_ahead', []), "gap_to_ahead")
            
            # Extract DRS zones
            drs_zones = []
            if 'zone_info' in raw_seq and isinstance(raw_seq['zone_info'], dict):
                zone_info = raw_seq['zone_info']
                if 'drs_zones' in zone_info:
                    drs_zones = zone_info['drs_zones']
            elif 'drs_zones' in raw_seq:
                drs_zones = raw_seq['drs_zones']
            elif zone_count > 0:
                drs_zones = list(range(1, zone_count + 1))
            
            if not isinstance(drs_zones, list):
                drs_zones = []
            
            # Skip if no speed data
            if len(speed_trace) < 10:
                if DEBUG_MODE:
                    print(f"   ⚠️  Skipping sequence {index}: insufficient speed data ({len(speed_trace)} points)")
                return None
            
            # Create sequence
            sequence = DRSSequence(
                sequence_id=sequence_id,
                circuit=circuit,
                session=session,
                lap_number=lap_number,
                driver=driver,
                year=year,
                zone_index=zone_index,
                zone_count=zone_count,
                speed_trace=speed_trace,
                throttle_trace=throttle_trace,
                brake_trace=brake_trace,
                ers_deployment=ers_deployment,
                gap_to_ahead=gap_to_ahead,
                position=position,
                drs_zones=drs_zones,
                label=label,
                sequence_length=len(speed_trace)
            )
            
            # Calculate quality score
            sequence.data_quality_score = self._calculate_quality_score(sequence)
            
            if DEBUG_MODE and index < 3:
                print(f"   ✅ Processed: {len(speed_trace)} speed points, label={label}, quality={sequence.data_quality_score:.2f}")
            
            return sequence
            
        except Exception as e:
            error_info = {
                'sequence_index': index,
                'error': str(e),
                'sequence_keys': list(raw_seq.keys()) if isinstance(raw_seq, dict) else str(type(raw_seq))
            }
            self.debug_info['processing_errors'].append(error_info)
            
            if DEBUG_MODE:
                print(f"   ❌ Error processing sequence {index}: {e}")
                if SAMPLE_PROBLEMATIC_SEQUENCES and len(self.debug_info['problematic_sequences']) < 5:
                    self.debug_info['problematic_sequences'].append({
                        'index': index,
                        'error': str(e),
                        'sample_data': {k: str(v)[:100] for k, v in raw_seq.items()} if isinstance(raw_seq, dict) else str(raw_seq)[:200]
                    })
            
            return None
    
    def _calculate_quality_score(self, sequence: DRSSequence) -> float:
        """Calculate data quality score"""
        score = 1.0
        
        # Length scoring
        if sequence.sequence_length >= 50:
            score *= 1.0
        elif sequence.sequence_length >= 30:
            score *= 0.9
        else:
            score *= 0.7
        
        # Data completeness
        if sequence.throttle_trace:
            score *= 1.05
        if sequence.brake_trace:
            score *= 1.05
        if sequence.ers_deployment:
            score *= 1.1
        if sequence.gap_to_ahead:
            score *= 1.1
        if sequence.drs_zones:
            score *= 1.05
        
        return min(1.0, score)
    
    def process_all_sequences(self, raw_sequences: List[Dict]) -> List[DRSSequence]:
        """Process all sequences with progress tracking"""
        processed_sequences = []
        
        print(f"🔄 Processing {len(raw_sequences)} sequences...")
        
        for i, raw_seq in enumerate(raw_sequences):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{len(raw_sequences)} ({i/len(raw_sequences)*100:.1f}%)")
            
            sequence = self.process_sequence_safely(raw_seq, i)
            if sequence:
                processed_sequences.append(sequence)
        
        print(f"✅ Successfully processed {len(processed_sequences)} sequences")
        print(f"⚠️  Failed to process {len(raw_sequences) - len(processed_sequences)} sequences")
        
        if self.debug_info['processing_errors']:
            print(f"🔍 Processing errors: {len(self.debug_info['processing_errors'])}")
            
            # Show common error types
            error_types = {}
            for error in self.debug_info['processing_errors']:
                error_type = error['error'].split(':')[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"   {error_type}: {count} times")
        
        return processed_sequences
    
    def create_features_and_metadata(self, sequences: List[DRSSequence]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create feature matrix and metadata"""
        print(f"🔧 Creating features from {len(sequences)} sequences...")
        
        feature_rows = []
        metadata_rows = []
        
        for sequence in sequences:
            if len(sequence.speed_trace) < 5:
                continue
            
            speed = np.array(sequence.speed_trace)
            
            # Basic features
            features = {
                'sequence_id': sequence.sequence_id,
                'circuit': sequence.circuit,
                'driver': sequence.driver,
                'max_speed': float(np.max(speed)),
                'min_speed': float(np.min(speed)),
                'avg_speed': float(np.mean(speed)),
                'speed_std': float(np.std(speed)),
                'speed_range': float(np.max(speed) - np.min(speed)),
                'sequence_length': sequence.sequence_length,
                'zone_count': sequence.zone_count,
                'drs_zone_count': len(sequence.drs_zones),
                'data_quality_score': sequence.data_quality_score,
                'year': sequence.year,
                'label': sequence.label
            }
            
            # Additional telemetry features
            if sequence.throttle_trace:
                throttle = np.array(sequence.throttle_trace)
                features['avg_throttle'] = float(np.mean(throttle))
                features['max_throttle'] = float(np.max(throttle))
            
            if sequence.brake_trace:
                brake = np.array(sequence.brake_trace)
                features['avg_brake'] = float(np.mean(brake))
                features['max_brake'] = float(np.max(brake))
            
            if sequence.ers_deployment:
                ers = np.array(sequence.ers_deployment)
                features['ers_avg'] = float(np.mean(ers))
                features['ers_max'] = float(np.max(ers))
            
            feature_rows.append(features)
            
            # Metadata
            metadata = {
                'sequence_id': sequence.sequence_id,
                'circuit': sequence.circuit,
                'session': sequence.session,
                'lap_number': sequence.lap_number,
                'driver': sequence.driver,
                'year': sequence.year,
                'position': sequence.position,
                'sequence_length': sequence.sequence_length,
                'data_quality_score': sequence.data_quality_score,
                'label': sequence.label,
                'is_labeled': sequence.label != -1,
                'needs_labeling': sequence.label == -1
            }
            metadata_rows.append(metadata)
        
        feature_df = pd.DataFrame(feature_rows)
        metadata_df = pd.DataFrame(metadata_rows)
        
        print(f"✅ Created feature matrix: {feature_df.shape}")
        return feature_df, metadata_df
    
    def save_debug_info(self, output_dir: str):
        """Save debugging information"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        debug_file = output_path / "debug_info.json"
        with open(debug_file, 'w') as f:
            json.dump(self.debug_info, f, indent=2)
        
        print(f"🔍 Debug info saved to {debug_file}")

def main():
    """Main processing function"""
    print("🏎️  F1 2023 Robust Data Processor")
    print("=" * 50)
    print("Processing with enhanced error handling...")
    print("=" * 50)
    
    # Initialize processor
    processor = F1RobustProcessor(DATA_PATH)
    
    # Load data
    raw_sequences = processor.load_data_safely()
    
    if not raw_sequences:
        print("❌ No sequences loaded")
        return
    
    # Process sequences
    sequences = processor.process_all_sequences(raw_sequences)
    
    if not sequences:
        print("❌ No sequences processed successfully")
        print("🔍 Check debug_info.json for detailed error analysis")
        processor.save_debug_info(OUTPUT_PATH)
        return
    
    # Create features
    feature_df, metadata_df = processor.create_features_and_metadata(sequences)
    
    # Create output directory
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save data
    feature_df.to_csv(output_path / '2023_features.csv', index=False)
    metadata_df.to_csv(output_path / '2023_metadata.csv', index=False)
    
    with open(output_path / '2023_sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save debug info
    processor.save_debug_info(OUTPUT_PATH)
    
    # Print results
    print("\n" + "="*50)
    print("🏁 PROCESSING COMPLETE!")
    print("="*50)
    
    labeled_count = (metadata_df['label'] != -1).sum()
    
    print(f"\n📊 RESULTS:")
    print(f"   Total sequences: {len(sequences)}")
    print(f"   Circuits: {metadata_df['circuit'].nunique()}")
    print(f"   Labeled sequences: {labeled_count}")
    print(f"   Average quality: {metadata_df['data_quality_score'].mean():.3f}")
    
    if labeled_count > 0:
        print(f"\n🎯 LABEL DISTRIBUTION:")
        label_dist = metadata_df[metadata_df['label'] != -1]['label'].value_counts().sort_index()
        for label, count in label_dist.items():
            print(f"   Label {label}: {count} sequences")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   2023_features.csv")
    print(f"   2023_metadata.csv") 
    print(f"   2023_sequences.pkl")
    print(f"   debug_info.json")
    
    if len(sequences) > 1000:
        print(f"\n🚀 SUCCESS! Ready for transformer training!")
    else:
        print(f"\n⚠️  Check debug_info.json for processing issues")

if __name__ == "__main__":
    main()