"""
Created on Sat Jun 22 10:31:55 2025

@author: sid
F1 2024 Fixed Data Processor
Fixes JSON serialization issue and extracts rich multi-dimensional labels
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

DATA_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/F1_2024_DRS_Dataset"  # Update this path
OUTPUT_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2024"
DEBUG_MODE = True
COMPARE_WITH_2023 = True



def safe_convert_to_int(value: Any, default: int = 0, field_name: str = "unknown") -> int:
    """Safely convert any value to integer (learned from 2023 issues)"""
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return int(value)
    
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            if DEBUG_MODE:
                print(f"   ⚠️  {field_name}: Can't convert string '{value}' to int, using {default}")
            return default
    
    if isinstance(value, dict):
        if DEBUG_MODE:
            print(f"   🔍 {field_name}: Got dict {list(value.keys())[:3]}..., extracting...")
        if 'value' in value:
            return safe_convert_to_int(value['value'], default, f"{field_name}.value")
        elif len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_int(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_int(value[0], default, f"{field_name}[0]")
    
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
    
    if isinstance(data, list):
        try:
            return [float(x) for x in data if x is not None]
        except (ValueError, TypeError):
            return []
    
    if isinstance(data, dict):
        possible_keys = ['data', 'values', 'trace', 'series', field_name.split('_')[-1]]
        
        for key in possible_keys:
            if key in data:
                return extract_telemetry_data(data[key], f"{field_name}.{key}")
        
        for key, value in data.items():
            if isinstance(value, list):
                return extract_telemetry_data(value, f"{field_name}.{key}")
    
    return []

def extract_rich_2024_labels(label_data: Any) -> Dict[str, Any]:
    """Extract rich multi-dimensional labels from 2024 format"""
    
    labels = {
        'overtake_decision': -1,
        'overtake_success': -1,
        'multi_zone_strategy': -1,
        'original_label': -1,
        'label_confidence': 0.0,
        'label_source': 'unknown'
    }
    
    if label_data is None:
        return labels
    
    # Handle dictionary labels (2024 rich format)
    if isinstance(label_data, dict):
        labels['label_source'] = 'rich_2024'
        
        # Extract overtake decision
        if 'overtake_decision' in label_data:
            decision = label_data['overtake_decision']
            labels['overtake_decision'] = safe_convert_to_int(decision, -1, 'overtake_decision')
        
        # Extract success indicator
        if 'overtake_success' in label_data:
            success = label_data['overtake_success']
            if isinstance(success, bool):
                labels['overtake_success'] = 1 if success else 0
            else:
                labels['overtake_success'] = safe_convert_to_int(success, -1, 'overtake_success')
        
        # Extract multi-zone strategy
        if 'multi_zone_strategy' in label_data:
            strategy = label_data['multi_zone_strategy']
            labels['multi_zone_strategy'] = safe_convert_to_int(strategy, -1, 'multi_zone_strategy')
        
        # Calculate confidence based on how many fields are filled
        filled_fields = sum(1 for k, v in labels.items() if k.endswith('_decision') or k.endswith('_success') or k.endswith('_strategy') if v != -1)
        labels['label_confidence'] = filled_fields / 3.0
        
        # Set original label for backward compatibility
        if labels['overtake_decision'] != -1:
            labels['original_label'] = labels['overtake_decision']
    
    # Handle simple labels (fallback)
    elif isinstance(label_data, (int, float)):
        labels['original_label'] = int(label_data)
        labels['overtake_decision'] = int(label_data)
        labels['label_confidence'] = 1.0
        labels['label_source'] = 'simple_int'
    
    elif isinstance(label_data, str) and label_data.isdigit():
        labels['original_label'] = int(label_data)
        labels['overtake_decision'] = int(label_data)
        labels['label_confidence'] = 1.0
        labels['label_source'] = 'simple_str'
    
    return labels

def convert_sets_to_lists(obj):
    """Convert sets to lists for JSON serialization"""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj

# ================================
# DATA CLASSES
# ================================

@dataclass
class DRSSequence2024Enhanced:
    """Enhanced data class for F1 2024 DRS sequence with rich labels"""
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
    
    # Enhanced labels (2024 format)
    overtake_decision: int = -1
    overtake_success: int = -1
    multi_zone_strategy: int = -1
    original_label: int = -1
    label_confidence: float = 0.0
    label_source: str = "unknown"
    
    # Computed features
    sequence_length: int = 0
    data_quality_score: float = 0.0
    china_gp_indicator: bool = False
    data_format_version: str = "2024_enhanced"

class F1_2024_EnhancedProcessor:
    """Enhanced processor for F1 2024 data with rich label extraction"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.year = 2024
        self.china_gp_sequences = 0
        self.rich_labels_found = 0
        self.simple_labels_found = 0
        
        # Use lists instead of sets to avoid JSON serialization issues
        self.debug_info = {
            'china_gp_found': False,
            'new_circuits': [],
            'format_changes': [],
            'processing_errors': [],
            'circuits_found': [],  # Changed from set to list
            'new_field_formats': [],  # Changed from set to list
            'returning_circuits': [],
            'rich_labels_found': 0,
            'simple_labels_found': 0
        }
        
        self.expected_2024_circuits = [
            'bahrain', 'saudi_arabia', 'australia', 'japan', 'china',
            'miami', 'monaco', 'canada', 'spain', 'austria', 'great_britain',
            'hungary', 'belgium', 'netherlands', 'italy', 'azerbaijan',
            'singapore', 'united_states', 'mexico', 'brazil', 'las_vegas',
            'qatar', 'abu_dhabi'
        ]
    
    def detect_2024_specific_features(self, raw_sequences: List[Dict]) -> Dict:
        """Detect 2024-specific features with proper list handling"""
        
        print("🔍 Detecting 2024-specific features...")
        
        circuits_found = []
        new_field_formats = []
        china_gp_sequences = 0
        
        for seq in raw_sequences[:100]:
            if isinstance(seq, dict):
                circuit = safe_convert_to_string(seq.get('circuit', ''), 'unknown', 'circuit').lower()
                if circuit not in circuits_found:
                    circuits_found.append(circuit)
                
                # Check for China GP
                if circuit == 'china':
                    china_gp_sequences += 1
                    if china_gp_sequences == 1:  # First China GP sequence found
                        print(f"   🎯 China GP sequence detected!")
                
                # Check for new field formats
                for key, value in seq.items():
                    if isinstance(value, dict) and key not in ['telemetry', 'context', 'zone_info']:
                        format_desc = f"{key}:{type(value).__name__}"
                        if format_desc not in new_field_formats:
                            new_field_formats.append(format_desc)
        
        # Update debug info
        self.debug_info['circuits_found'] = sorted(circuits_found)
        self.debug_info['new_field_formats'] = new_field_formats
        self.debug_info['china_gp_found'] = china_gp_sequences > 0
        
        # Determine returning circuits
        returning_circuits = []
        for circuit in ['china']:  # Known returning circuits
            if circuit in circuits_found:
                returning_circuits.append(circuit)
        
        self.debug_info['returning_circuits'] = returning_circuits
        
        print(f"   📊 Circuits found: {circuits_found}")
        print(f"   🔄 Returning circuits: {returning_circuits}")
        print(f"   🆕 New field formats: {new_field_formats}")
        
        return self.debug_info
    
    def load_2024_data(self) -> List[Dict]:
        """Load 2024 data with format detection"""
        all_sequences = []
        
        print(f"🔄 Loading 2024 F1 data from {self.data_path}...")
        
        # Look for JSON files
        json_files = list(self.data_path.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.data_path}")
            return []
        
        print(f"📁 Found {len(json_files)} JSON files: {[f.name for f in json_files]}")
        
        # Process each file
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"📄 Processing {file_path.name} ({file_size:.1f}MB)...")
                
                sequences = self._extract_sequences_from_file(data, file_path.name)
                if sequences:
                    all_sequences.extend(sequences)
                    print(f"   ✅ Extracted {len(sequences)} sequences")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
                self.debug_info['processing_errors'].append(f"{file_path.name}: {str(e)}")
        
        # Detect features
        if all_sequences:
            self.detect_2024_specific_features(all_sequences)
        
        print(f"📊 Total 2024 sequences loaded: {len(all_sequences)}")
        return all_sequences
    
    def _extract_sequences_from_file(self, data: Any, filename: str) -> List[Dict]:
        """Extract sequences from various file structures"""
        sequences = []
        
        if isinstance(data, list):
            for seq in data:
                if isinstance(seq, dict):
                    seq['source_file'] = filename
                    seq['year'] = 2024
                    sequences.append(seq)
        
        elif isinstance(data, dict):
            if 'sequences' in data and isinstance(data['sequences'], list):
                for seq in data['sequences']:
                    if isinstance(seq, dict):
                        seq['source_file'] = filename
                        seq['year'] = 2024
                        sequences.append(seq)
            else:
                for key, value in data.items():
                    if isinstance(value, dict):
                        circuit_sequences = self._extract_from_circuit_data(value, key, filename)
                        sequences.extend(circuit_sequences)
        
        return sequences
    
    def _extract_from_circuit_data(self, circuit_data: Dict, circuit_key: str, filename: str) -> List[Dict]:
        """Extract sequences from circuit-based data structure"""
        sequences = []
        
        def search_recursively(obj, path=""):
            if isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) and self._looks_like_sequence(item):
                        item['circuit'] = circuit_key.split('_')[0].lower()
                        item['source_file'] = filename
                        item['year'] = 2024
                        sequences.append(item)
                    else:
                        search_recursively(item, f"{path}[{i}]")
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    search_recursively(value, f"{path}.{key}")
        
        search_recursively(circuit_data)
        return sequences
    
    def _looks_like_sequence(self, obj: Dict) -> bool:
        """Check if object looks like an F1 sequence"""
        required_indicators = ['circuit', 'driver'] 
        telemetry_indicators = ['telemetry', 'speed', 'speed_trace']
        
        has_required = any(indicator in obj for indicator in required_indicators)
        has_telemetry = any(indicator in obj for indicator in telemetry_indicators)
        
        return has_required or has_telemetry
    
    def process_2024_sequence(self, raw_seq: Dict, index: int) -> Optional[DRSSequence2024Enhanced]:
        """Process a single 2024 sequence with enhanced label extraction"""
        
        if DEBUG_MODE and index < 3:
            print(f"\n🔍 Debug 2024 sequence {index}:")
            print(f"   Keys: {list(raw_seq.keys())}")
            
            # Check label structure
            if 'label' in raw_seq:
                label = raw_seq['label']
                if isinstance(label, dict):
                    print(f"   🎯 Rich label detected: {list(label.keys())}")
                else:
                    print(f"   📝 Simple label: {label}")
        
        try:
            # Extract basic metadata
            sequence_id = safe_convert_to_string(raw_seq.get('sequence_id', f"2024_seq_{index}"), f"2024_seq_{index}", "sequence_id")
            circuit = safe_convert_to_string(raw_seq.get('circuit', 'unknown'), 'unknown', "circuit").lower()
            session = safe_convert_to_string(raw_seq.get('session', 'race'), 'race', "session")
            lap_number = safe_convert_to_float(raw_seq.get('lap_number', 0), 0.0, "lap_number")
            driver = safe_convert_to_string(raw_seq.get('driver', 'unknown'), 'unknown', "driver")
            year = safe_convert_to_int(raw_seq.get('year', 2024), 2024, "year")
            zone_index = safe_convert_to_int(raw_seq.get('zone_index', 0), 0, "zone_index")
            zone_count = safe_convert_to_int(raw_seq.get('zone_count', 0), 0, "zone_count")
            position = safe_convert_to_int(raw_seq.get('position', 99), 99, "position")
            
            # Extract rich labels
            label_data = raw_seq.get('label', None)
            rich_labels = extract_rich_2024_labels(label_data)
            
            # Track label types
            if rich_labels['label_source'] == 'rich_2024':
                self.rich_labels_found += 1
            else:
                self.simple_labels_found += 1
            
            # Track China GP
            china_gp_indicator = (circuit == 'china')
            if china_gp_indicator:
                self.china_gp_sequences += 1
            
            # Extract telemetry
            speed_trace = []
            throttle_trace = []
            brake_trace = []
            ers_deployment = []
            gap_to_ahead = []
            
            if 'telemetry' in raw_seq:
                telemetry_data = raw_seq['telemetry']
                if isinstance(telemetry_data, dict):
                    speed_trace = extract_telemetry_data(telemetry_data.get('speed', []), "speed")
                    throttle_trace = extract_telemetry_data(telemetry_data.get('throttle', []), "throttle")
                    brake_trace = extract_telemetry_data(telemetry_data.get('brake', []), "brake")
                    ers_deployment = extract_telemetry_data(telemetry_data.get('ers', []), "ers")
                    gap_to_ahead = extract_telemetry_data(telemetry_data.get('gap_ahead', []), "gap_ahead")
            
            # Fallback to top-level
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
            
            # Skip if insufficient data
            if len(speed_trace) < 10:
                return None
            
            # Create enhanced sequence
            sequence = DRSSequence2024Enhanced(
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
                overtake_decision=rich_labels['overtake_decision'],
                overtake_success=rich_labels['overtake_success'],
                multi_zone_strategy=rich_labels['multi_zone_strategy'],
                original_label=rich_labels['original_label'],
                label_confidence=rich_labels['label_confidence'],
                label_source=rich_labels['label_source'],
                sequence_length=len(speed_trace),
                china_gp_indicator=china_gp_indicator
            )
            
            # Calculate quality
            sequence.data_quality_score = self._calculate_quality_score(sequence)
            
            return sequence
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"   ❌ Error processing 2024 sequence {index}: {e}")
            self.debug_info['processing_errors'].append(f"Sequence {index}: {str(e)}")
            return None
    
    def _calculate_quality_score(self, sequence: DRSSequence2024Enhanced) -> float:
        """Calculate quality score for 2024 enhanced data"""
        score = 1.0
        
        # Length scoring
        if sequence.sequence_length >= 50:
            score *= 1.0
        elif sequence.sequence_length >= 30:
            score *= 0.9
        else:
            score *= 0.7
        
        # Data completeness bonuses
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
        
        # Rich label bonus
        if sequence.label_confidence > 0.5:
            score *= 1.1
        
        # China GP bonus
        if sequence.china_gp_indicator:
            score *= 1.02
        
        return min(1.0, score)
    
    def process_all_2024_sequences(self, raw_sequences: List[Dict]) -> List[DRSSequence2024Enhanced]:
        """Process all 2024 sequences with enhanced labels"""
        processed_sequences = []
        
        print(f"🔄 Processing {len(raw_sequences)} 2024 sequences with enhanced label extraction...")
        
        for i, raw_seq in enumerate(raw_sequences):
            if i % 1000 == 0 and i > 0:
                print(f"   Progress: {i}/{len(raw_sequences)} ({i/len(raw_sequences)*100:.1f}%)")
            
            sequence = self.process_2024_sequence(raw_seq, i)
            if sequence:
                processed_sequences.append(sequence)
        
        # Update debug info
        self.debug_info['rich_labels_found'] = self.rich_labels_found
        self.debug_info['simple_labels_found'] = self.simple_labels_found
        
        print(f"✅ Successfully processed {len(processed_sequences)} 2024 sequences")
        print(f"🎯 China GP sequences: {self.china_gp_sequences}")
        print(f"📊 Rich labels found: {self.rich_labels_found}")
        print(f"📊 Simple labels found: {self.simple_labels_found}")
        
        return processed_sequences
    
    def create_enhanced_features_and_metadata(self, sequences: List[DRSSequence2024Enhanced]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create enhanced feature matrix and metadata for 2024"""
        print(f"🔧 Creating enhanced 2024 features from {len(sequences)} sequences...")
        
        feature_rows = []
        metadata_rows = []
        
        for sequence in sequences:
            if len(sequence.speed_trace) < 5:
                continue
            
            speed = np.array(sequence.speed_trace)
            
            # Enhanced features with rich labels
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
                'china_gp': sequence.china_gp_indicator,
                # Rich label features
                'overtake_decision': sequence.overtake_decision,
                'overtake_success': sequence.overtake_success,
                'multi_zone_strategy': sequence.multi_zone_strategy,
                'original_label': sequence.original_label,
                'label_confidence': sequence.label_confidence,
                'label_source': sequence.label_source
            }
            
            # Telemetry features
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
            
            # Enhanced metadata
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
                'china_gp': sequence.china_gp_indicator,
                'data_format_version': sequence.data_format_version,
                # Rich labels
                'overtake_decision': sequence.overtake_decision,
                'overtake_success': sequence.overtake_success,
                'multi_zone_strategy': sequence.multi_zone_strategy,
                'original_label': sequence.original_label,
                'label_confidence': sequence.label_confidence,
                'label_source': sequence.label_source,
                'is_labeled': sequence.overtake_decision != -1,
                'needs_labeling': sequence.overtake_decision == -1,
                'has_rich_labels': sequence.label_source == 'rich_2024'
            }
            metadata_rows.append(metadata)
        
        feature_df = pd.DataFrame(feature_rows)
        metadata_df = pd.DataFrame(metadata_rows)
        
        print(f"✅ Created enhanced 2024 feature matrix: {feature_df.shape}")
        
        return feature_df, metadata_df

def main():
    """Main enhanced 2024 processing function"""
    print("🏎️  F1 2024 Enhanced Data Processor")
    print("=" * 50)
    print("Key Features:")
    print("• China GP return detection")
    print("• Rich multi-dimensional label extraction")
    print("• Fixed JSON serialization")
    print("• Enhanced error handling")
    print("=" * 50)
    
    # Check data path
    if not Path(DATA_PATH).exists():
        print(f"❌ Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH variable")
        return
    
    # Initialize processor
    processor = F1_2024_EnhancedProcessor(DATA_PATH)
    
    # Load data
    raw_sequences = processor.load_2024_data()
    
    if not raw_sequences:
        print("❌ No 2024 sequences loaded")
        return
    
    # Process sequences
    sequences = processor.process_all_2024_sequences(raw_sequences)
    
    if not sequences:
        print("❌ No 2024 sequences processed successfully")
        return
    
    # Create features and metadata
    feature_df, metadata_df = processor.create_enhanced_features_and_metadata(sequences)
    
    # Save data
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    feature_df.to_csv(output_path / '2024_features_enhanced.csv', index=False)
    metadata_df.to_csv(output_path / '2024_metadata_enhanced.csv', index=False)
    
    with open(output_path / '2024_sequences_enhanced.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save debug info (with proper JSON serialization)
    debug_info_clean = convert_sets_to_lists(processor.debug_info)
    with open(output_path / '2024_debug_info_fixed.json', 'w') as f:
        json.dump(debug_info_clean, f, indent=2)
    
    # Print results
    print("\n" + "="*50)
    print("🏁 F1 2024 ENHANCED PROCESSING COMPLETE!")
    print("="*50)
    
    labeled_count = (metadata_df['overtake_decision'] != -1).sum()
    rich_labeled_count = (metadata_df['label_source'] == 'rich_2024').sum()
    
    print(f"\n📊 2024 ENHANCED RESULTS:")
    print(f"   Total sequences: {len(sequences):,}")
    print(f"   Circuits: {metadata_df['circuit'].nunique()}")
    print(f"   Labeled sequences: {labeled_count:,}")
    print(f"   Rich labels: {rich_labeled_count:,} ({rich_labeled_count/len(metadata_df)*100:.1f}%)")
    print(f"   Average quality: {metadata_df['data_quality_score'].mean():.3f}")
    print(f"   Average label confidence: {metadata_df['label_confidence'].mean():.3f}")
    
    # 2024 highlights
    print(f"\n🎯 2024 HIGHLIGHTS:")
    if processor.china_gp_sequences > 0:
        print(f"   ✅ China GP return: {processor.china_gp_sequences:,} sequences")
        china_quality = metadata_df[metadata_df['china_gp'] == True]['data_quality_score'].mean()
        print(f"   📊 China GP quality: {china_quality:.3f}")
    
    circuits_found = sorted(metadata_df['circuit'].unique())
    print(f"   🏁 Circuits: {', '.join(circuits_found)}")
    
    # Rich label analysis
    if rich_labeled_count > 0:
        print(f"\n🎯 RICH LABEL ANALYSIS:")
        
        # Overtake decision distribution
        decision_dist = metadata_df[metadata_df['overtake_decision'] != -1]['overtake_decision'].value_counts().sort_index()
        if len(decision_dist) > 0:
            print(f"   Overtake Decisions:")
            for decision, count in decision_dist.items():
                print(f"      Decision {decision}: {count:,} sequences")
        
        # Success distribution
        success_dist = metadata_df[metadata_df['overtake_success'] != -1]['overtake_success'].value_counts().sort_index()
        if len(success_dist) > 0:
            print(f"   Success Outcomes:")
            for success, count in success_dist.items():
                outcome = "Success" if success == 1 else "Failure" if success == 0 else f"Value {success}"
                print(f"      {outcome}: {count:,} sequences")
        
        # Strategy distribution
        strategy_dist = metadata_df[metadata_df['multi_zone_strategy'] != -1]['multi_zone_strategy'].value_counts().sort_index()
        if len(strategy_dist) > 0:
            print(f"   Multi-Zone Strategies:")
            for strategy, count in strategy_dist.items():
                print(f"      Strategy {strategy}: {count:,} sequences")
    
    print(f"\n📁 ENHANCED FILES CREATED:")
    print(f"   2024_features_enhanced.csv ({feature_df.shape})")
    print(f"   2024_metadata_enhanced.csv")
    print(f"   2024_sequences_enhanced.pkl")
    print(f"   2024_debug_info_fixed.json")
    
    print(f"\n🚀 ENHANCED CAPABILITIES:")
    print(f"   ✅ Multi-dimensional labels extracted")
    print(f"   ✅ JSON serialization fixed")
    print(f"   ✅ Rich overtaking behavior analysis ready")
    print(f"   ✅ Advanced model training data prepared")

if __name__ == "__main__":
    main()