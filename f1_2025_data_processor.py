"""
Created on Sat Jun 22 13:20:20 2025

@author: sid
F1 2025 Austrian GP DRS Decision AI Processor
Preprocesses 2025 data (Bahrain-Canada) to predict DRS strategies for Austrian GP
Includes transfer learning from 2024 Austrian GP data and circuit-specific features

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

DATA_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/F1_2025_Partial_Dataset"  # Update this path
OUTPUT_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2025_austria_prediction"
AUSTRIA_2024_DATA_PATH = "/Users/sid/Downloads/F1-DRS-Decision-AI/processed_data/2024"  # For transfer learning
DEBUG_MODE = True
PREPARE_FOR_AUSTRIA = True
GENERATE_SYNTHETIC_LABELS = False  # Set True if no labels exist

# ================================
# 2025 AUSTRIAN GP PREDICTION CONFIGURATION
# ================================

# Available 2025 circuits (Bahrain through Canada)
AVAILABLE_2025_CIRCUITS = [
    'bahrain',      # Round 1 - March 2
    'saudi_arabia', # Round 2 - March 9  
    'australia',    # Round 3 - March 16
    'japan',        # Round 4 - April 6
    'china',        # Round 5 - April 20
    'miami',        # Round 6 - May 4
    'monaco',       # Round 7 - May 25
    'canada'        # Round 8 - June 8
]

# TARGET: Austria GP - June 29, 2025 (Round 9)
TARGET_CIRCUIT = 'austria'

# Austrian GP Track Characteristics (Red Bull Ring)
AUSTRIA_TRACK_PROFILE = {
    'track_length_km': 4.318,
    'lap_record_speed': 307.0,  # km/h from DRS zones
    'elevation_change': 65,  # meters
    'drs_zones': 3,
    'main_straight_length': 1.2,  # km
    'sector_1_technical': True,
    'sector_2_elevation': True,
    'sector_3_high_speed': True,
    'overtaking_difficulty': 'medium',
    'typical_race_pace': 245.0,  # Average race speed
    'drs_effectiveness': 'high',
    'slipstream_benefit': 'very_high'
}

# Circuit similarity for transfer learning (based on track characteristics)
CIRCUIT_SIMILARITY_TO_AUSTRIA = {
    'bahrain': 0.7,      # Similar high-speed sections and DRS effectiveness
    'saudi_arabia': 0.6, # High-speed but different layout
    'australia': 0.5,    # Some similar characteristics
    'japan': 0.4,        # Technical but different
    'china': 0.8,        # Good straight-line speed similarity
    'miami': 0.6,        # Modern circuit with good DRS zones
    'monaco': 0.2,       # Completely different (low-speed, no DRS effectiveness)
    'canada': 0.9        # Very similar: high-speed straights, good DRS zones
}

def safe_convert_to_int(value: Any, default: int = 0, field_name: str = "unknown") -> int:
    """Safely convert any value to integer with enhanced 2025 handling"""
    if value is None or value == "" or value == "null":
        return default
    
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return int(value)
    
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ['null', 'none', 'nan']:
            return default
        try:
            return int(float(value))
        except ValueError:
            if DEBUG_MODE:
                print(f"   ⚠️  {field_name}: Can't convert string '{value}' to int, using {default}")
            return default
    
    if isinstance(value, dict):
        if DEBUG_MODE:
            print(f"   🔍 {field_name}: Got dict {list(value.keys())[:3]}..., extracting...")
        
        # Try multiple possible keys
        possible_keys = ['value', 'data', 'result', 'measurement', 'reading']
        for key in possible_keys:
            if key in value:
                return safe_convert_to_int(value[key], default, f"{field_name}.{key}")
        
        if len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_int(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_int(value[0], default, f"{field_name}[0]")
    
    return default

def safe_convert_to_float(value: Any, default: float = 0.0, field_name: str = "unknown") -> float:
    """Safely convert any value to float with enhanced 2025 handling"""
    if value is None or value == "" or value == "null":
        return default
    
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return float(value)
    
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ['null', 'none', 'nan']:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    if isinstance(value, dict):
        possible_keys = ['value', 'data', 'result', 'measurement', 'reading']
        for key in possible_keys:
            if key in value:
                return safe_convert_to_float(value[key], default, f"{field_name}.{key}")
        
        if len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_float(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_float(value[0], default, f"{field_name}[0]")
    
    return default

def safe_convert_to_string(value: Any, default: str = "unknown", field_name: str = "unknown") -> str:
    """Safely convert any value to string"""
    if value is None or value == "" or value == "null":
        return default
    
    if isinstance(value, str):
        value = value.strip()
        return value if value else default
    
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return default
        return str(value)
    
    if isinstance(value, dict):
        possible_keys = ['value', 'name', 'label', 'text', 'description']
        for key in possible_keys:
            if key in value:
                return safe_convert_to_string(value[key], default, f"{field_name}.{key}")
        
        if len(value) == 1:
            key = list(value.keys())[0]
            return safe_convert_to_string(value[key], default, f"{field_name}.{key}")
        return default
    
    if isinstance(value, list) and len(value) > 0:
        return safe_convert_to_string(value[0], default, f"{field_name}[0]")
    
    return str(value)

def extract_enhanced_telemetry_data(data: Any, field_name: str = "telemetry") -> List[float]:
    """Enhanced telemetry extraction for 2025"""
    if not data:
        return []
    
    if isinstance(data, list):
        try:
            return [float(x) for x in data if x is not None and str(x).lower() not in ['nan', 'null']]
        except (ValueError, TypeError):
            return []
    
    if isinstance(data, dict):
        possible_keys = [
            'data', 'values', 'trace', 'series', 'measurements', 
            'readings', 'samples', 'points', field_name.split('_')[-1]
        ]
        
        for key in possible_keys:
            if key in data:
                result = extract_enhanced_telemetry_data(data[key], f"{field_name}.{key}")
                if result:
                    return result
        
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                result = extract_enhanced_telemetry_data(value, f"{field_name}.{key}")
                if result:
                    return result
    
    return []

def extract_fixed_2025_labels(label_data: Any) -> Dict[str, Any]:
    """
    FIXED label extraction for 2025 - addresses 2024 issue where all labels were -1
    Multiple fallback strategies to ensure we get usable labels
    """
    
    labels = {
        'overtake_decision': -1,
        'overtake_success': -1,
        'multi_zone_strategy': -1,
        'drs_activation': -1,
        'original_label': -1,
        'label_confidence': 0.0,
        'label_source': 'unknown'
    }
    
    if label_data is None:
        return labels
    
    # CRITICAL FIX: Handle simple numeric labels first (most reliable)
    if isinstance(label_data, (int, float)) and not np.isnan(label_data):
        val = int(label_data)
        labels['original_label'] = val
        labels['overtake_decision'] = val
        labels['drs_activation'] = val
        labels['label_confidence'] = 1.0
        labels['label_source'] = 'simple_numeric'
        if DEBUG_MODE:
            print(f"   ✅ Extracted numeric label: {val}")
        return labels
    
    # Handle string numeric labels
    if isinstance(label_data, str):
        label_str = label_data.strip()
        if label_str.isdigit():
            val = int(label_str)
            labels['original_label'] = val
            labels['overtake_decision'] = val
            labels['drs_activation'] = val
            labels['label_confidence'] = 1.0
            labels['label_source'] = 'string_numeric'
            if DEBUG_MODE:
                print(f"   ✅ Extracted string numeric label: {val}")
            return labels
        
        # Handle boolean strings
        elif label_str.lower() in ['true', 'yes', '1', 'activate', 'use']:
            labels['original_label'] = 1
            labels['overtake_decision'] = 1
            labels['drs_activation'] = 1
            labels['label_confidence'] = 1.0
            labels['label_source'] = 'boolean_true'
            return labels
        elif label_str.lower() in ['false', 'no', '0', 'deactivate', 'none']:
            labels['original_label'] = 0
            labels['overtake_decision'] = 0
            labels['drs_activation'] = 0
            labels['label_confidence'] = 1.0
            labels['label_source'] = 'boolean_false'
            return labels
    
    # Handle rich dictionary labels (2025 enhanced format)
    if isinstance(label_data, dict):
        labels['label_source'] = 'rich_2025'
        filled_count = 0
        
        # Try various field names for overtake decision
        decision_fields = [
            'overtake_decision', 'decision', 'drs_decision', 'action', 
            'strategy', 'choice', 'activation', 'use_drs'
        ]
        for field in decision_fields:
            if field in label_data:
                val = safe_convert_to_int(label_data[field], -1, f'label.{field}')
                if val != -1:
                    labels['overtake_decision'] = val
                    labels['original_label'] = val
                    filled_count += 1
                    break
        
        # Try various field names for success
        success_fields = [
            'overtake_success', 'success', 'outcome', 'result', 
            'effective', 'worked', 'achieved'
        ]
        for field in success_fields:
            if field in label_data:
                value = label_data[field]
                if isinstance(value, bool):
                    labels['overtake_success'] = 1 if value else 0
                    filled_count += 1
                elif isinstance(value, str) and value.lower() in ['true', 'success', 'yes']:
                    labels['overtake_success'] = 1
                    filled_count += 1
                elif isinstance(value, str) and value.lower() in ['false', 'failure', 'no']:
                    labels['overtake_success'] = 0
                    filled_count += 1
                else:
                    val = safe_convert_to_int(value, -1, f'label.{field}')
                    if val != -1:
                        labels['overtake_success'] = val
                        filled_count += 1
                break
        
        # Try DRS activation field
        drs_fields = ['drs_activation', 'drs_used', 'drs', 'activation']
        for field in drs_fields:
            if field in label_data:
                val = safe_convert_to_int(label_data[field], -1, f'label.{field}')
                if val != -1:
                    labels['drs_activation'] = val
                    filled_count += 1
                    break
        
        # Calculate confidence based on filled fields
        total_fields = 3  # decision, success, activation
        labels['label_confidence'] = filled_count / total_fields if filled_count > 0 else 0.0
        
        if DEBUG_MODE and filled_count > 0:
            print(f"   ✅ Extracted rich labels: {filled_count}/{total_fields} fields")
    
    # If still no labels and GENERATE_SYNTHETIC_LABELS is True
    if labels['overtake_decision'] == -1 and GENERATE_SYNTHETIC_LABELS:
        # This will be filled in by telemetry-based label generation
        labels['label_source'] = 'needs_synthetic'
    
    return labels

def generate_synthetic_labels_from_telemetry(speed_trace: List[float], 
                                           circuit: str,
                                           gap_to_ahead: List[float] = None) -> Dict[str, Any]:
    """Generate synthetic labels from telemetry patterns for Austrian GP training"""
    
    if len(speed_trace) < 10:
        return {
            'overtake_decision': -1,
            'label_confidence': 0.0,
            'label_source': 'insufficient_data'
        }
    
    speed = np.array(speed_trace)
    max_speed = np.max(speed)
    min_speed = np.min(speed)
    avg_speed = np.mean(speed)
    speed_range = max_speed - min_speed
    
    # Circuit-specific thresholds based on similarity to Austria
    similarity = CIRCUIT_SIMILARITY_TO_AUSTRIA.get(circuit, 0.5)
    
    # Adjust thresholds based on circuit similarity to Austria
    if similarity > 0.7:  # High similarity (Canada, China, Bahrain)
        speed_threshold = 280
        range_threshold = 45
        confidence_base = 0.8
    elif similarity > 0.5:  # Medium similarity 
        speed_threshold = 270
        range_threshold = 40
        confidence_base = 0.7
    else:  # Low similarity (Monaco)
        speed_threshold = 250
        range_threshold = 30
        confidence_base = 0.5
    
    # Decision logic for DRS usage
    if max_speed > speed_threshold and speed_range > range_threshold:
        decision = 1  # Likely DRS used
        confidence = confidence_base * 1.0
    elif max_speed > speed_threshold - 20 and speed_range > range_threshold - 15:
        decision = 1  # Probably DRS used
        confidence = confidence_base * 0.8
    elif max_speed < speed_threshold - 40 or speed_range < range_threshold - 20:
        decision = 0  # Likely no DRS
        confidence = confidence_base * 0.7
    else:
        decision = 1  # Default to DRS usage (F1 drivers use it when available)
        confidence = confidence_base * 0.6
    
    # Adjust confidence based on gap data if available
    if gap_to_ahead and len(gap_to_ahead) > 5:
        avg_gap = np.mean([g for g in gap_to_ahead if g > 0])
        if 0.5 <= avg_gap <= 2.0:  # Close racing, more likely to use DRS
            confidence *= 1.1
            decision = 1
        elif avg_gap > 5.0:  # Large gap, less strategic need
            confidence *= 0.9
    
    return {
        'overtake_decision': decision,
        'drs_activation': decision,
        'original_label': decision,
        'label_confidence': min(confidence, 1.0),
        'label_source': f'synthetic_{circuit}_sim{similarity:.1f}'
    }

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
class DRSSequence2025Austrian:
    """Enhanced data class for 2025 F1 DRS sequences optimized for Austrian GP prediction"""
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
    
    # Enhanced labels
    overtake_decision: int = -1
    overtake_success: int = -1
    multi_zone_strategy: int = -1
    drs_activation: int = -1
    original_label: int = -1
    label_confidence: float = 0.0
    label_source: str = "unknown"
    
    # Austrian GP specific features
    circuit_similarity_to_austria: float = 0.0
    austrian_gp_relevance: float = 0.0
    high_speed_sector_performance: float = 0.0
    drs_effectiveness_score: float = 0.0
    
    # Computed features
    sequence_length: int = 0
    data_quality_score: float = 0.0
    data_format_version: str = "2025_austrian_gp"

class F1_2025_AustrianGP_Processor:
    """2025 F1 data processor specifically designed for Austrian GP DRS prediction"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.year = 2025
        self.target_circuit = TARGET_CIRCUIT
        self.labeled_sequences = 0
        self.unlabeled_sequences = 0
        self.synthetic_labels_created = 0
        
        self.debug_info = {
            'austria_preparation': True,
            'target_circuit': TARGET_CIRCUIT,
            'available_circuits': [],
            'processing_errors': [],
            'label_sources': [],
            'austria_transfer_learning_ready': False,
            'labeled_sequences': 0,
            'synthetic_labels_created': 0,
            'circuit_similarity_scores': CIRCUIT_SIMILARITY_TO_AUSTRIA.copy()
        }
    
    def load_2025_data_for_austria_prediction(self) -> List[Dict]:
        """Load 2025 data specifically for Austrian GP prediction"""
        all_sequences = []
        
        print(f"🏎️  Loading 2025 F1 data for Austrian GP prediction...")
        print(f"🎯 Target: {TARGET_CIRCUIT.upper()} GP (Round 9)")
        print(f"📊 Training on: {', '.join([c.upper() for c in AVAILABLE_2025_CIRCUITS])}")
        
        json_files = list(self.data_path.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.data_path}")
            return []
        
        print(f"📁 Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"📄 Processing {file_path.name} ({file_size:.1f}MB)...")
                
                sequences = self._extract_sequences_from_file(data, file_path.name)
                if sequences:
                    # Filter only available circuits (up to Canada)
                    filtered_sequences = []
                    for seq in sequences:
                        circuit = safe_convert_to_string(seq.get('circuit', ''), 'unknown', 'circuit').lower()
                        if circuit in AVAILABLE_2025_CIRCUITS:
                            filtered_sequences.append(seq)
                        elif circuit == TARGET_CIRCUIT:
                            print(f"   ⚠️  Found {TARGET_CIRCUIT} data - this will be used for validation only")
                    
                    all_sequences.extend(filtered_sequences)
                    print(f"   ✅ Extracted {len(filtered_sequences)} relevant sequences")
                
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")
                self.debug_info['processing_errors'].append(f"{file_path.name}: {str(e)}")
        
        # Analyze available circuits
        circuits_found = []
        for seq in all_sequences:
            circuit = safe_convert_to_string(seq.get('circuit', ''), 'unknown', 'circuit').lower()
            if circuit not in circuits_found:
                circuits_found.append(circuit)
        
        self.debug_info['available_circuits'] = sorted(circuits_found)
        
        print(f"📊 Total 2025 sequences for Austria prediction: {len(all_sequences)}")
        print(f"🏁 Available circuits: {circuits_found}")
        
        return all_sequences
    
    def _extract_sequences_from_file(self, data: Any, filename: str) -> List[Dict]:
        """Extract sequences from various file structures"""
        sequences = []
        
        if isinstance(data, list):
            for seq in data:
                if isinstance(seq, dict):
                    seq['source_file'] = filename
                    seq['year'] = 2025
                    sequences.append(seq)
        
        elif isinstance(data, dict):
            if 'sequences' in data and isinstance(data['sequences'], list):
                for seq in data['sequences']:
                    if isinstance(seq, dict):
                        seq['source_file'] = filename
                        seq['year'] = 2025
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
                        item['year'] = 2025
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
        required_indicators = ['circuit', 'driver', 'lap_number'] 
        telemetry_indicators = ['telemetry', 'speed', 'speed_trace']
        
        has_required = any(indicator in obj for indicator in required_indicators)
        has_telemetry = any(indicator in obj for indicator in telemetry_indicators)
        
        return has_required or has_telemetry
    
    def process_2025_sequence_for_austria(self, raw_seq: Dict, index: int) -> Optional[DRSSequence2025Austrian]:
        """Process a 2025 sequence with Austrian GP prediction focus"""
        
        if DEBUG_MODE and index < 3:
            print(f"\n🔍 Debug 2025 sequence {index} for Austria prediction:")
            print(f"   Keys: {list(raw_seq.keys())}")
            
            if 'label' in raw_seq:
                label = raw_seq['label']
                print(f"   Label: {label} (type: {type(label)})")
        
        try:
            # Extract basic metadata
            sequence_id = safe_convert_to_string(raw_seq.get('sequence_id', f"2025_seq_{index}"), f"2025_seq_{index}", "sequence_id")
            circuit = safe_convert_to_string(raw_seq.get('circuit', 'unknown'), 'unknown', "circuit").lower()
            session = safe_convert_to_string(raw_seq.get('session', 'race'), 'race', "session")
            lap_number = safe_convert_to_float(raw_seq.get('lap_number', 0), 0.0, "lap_number")
            driver = safe_convert_to_string(raw_seq.get('driver', 'unknown'), 'unknown', "driver")
            year = safe_convert_to_int(raw_seq.get('year', 2025), 2025, "year")
            zone_index = safe_convert_to_int(raw_seq.get('zone_index', 0), 0, "zone_index")
            zone_count = safe_convert_to_int(raw_seq.get('zone_count', 0), 0, "zone_count")
            position = safe_convert_to_int(raw_seq.get('position', 99), 99, "position")
            
            # Skip if not from available circuits
            if circuit not in AVAILABLE_2025_CIRCUITS:
                return None
            
            # Extract labels with FIXED extraction
            label_data = raw_seq.get('label', None)
            labels = extract_fixed_2025_labels(label_data)
            
            # Extract telemetry
            speed_trace = []
            throttle_trace = []
            brake_trace = []
            ers_deployment = []
            gap_to_ahead = []
            
            if 'telemetry' in raw_seq:
                telemetry_data = raw_seq['telemetry']
                if isinstance(telemetry_data, dict):
                    speed_trace = extract_enhanced_telemetry_data(telemetry_data.get('speed', []), "speed")
                    throttle_trace = extract_enhanced_telemetry_data(telemetry_data.get('throttle', []), "throttle")
                    brake_trace = extract_enhanced_telemetry_data(telemetry_data.get('brake', []), "brake")
                    ers_deployment = extract_enhanced_telemetry_data(telemetry_data.get('ers', []), "ers")
                    gap_to_ahead = extract_enhanced_telemetry_data(telemetry_data.get('gap_ahead', []), "gap_ahead")
            
            # Fallback to top-level telemetry
            if not speed_trace:
                speed_trace = extract_enhanced_telemetry_data(raw_seq.get('speed_trace', []), "speed_trace")
            if not throttle_trace:
                throttle_trace = extract_enhanced_telemetry_data(raw_seq.get('throttle_trace', []), "throttle_trace")
            if not brake_trace:
                brake_trace = extract_enhanced_telemetry_data(raw_seq.get('brake_trace', []), "brake_trace")
            if not ers_deployment:
                ers_deployment = extract_enhanced_telemetry_data(raw_seq.get('ers_deployment', []), "ers_deployment")
            if not gap_to_ahead:
                gap_to_ahead = extract_enhanced_telemetry_data(raw_seq.get('gap_to_ahead', []), "gap_to_ahead")
            
            # Generate synthetic labels if needed and enabled
            if labels['overtake_decision'] == -1 and GENERATE_SYNTHETIC_LABELS and len(speed_trace) >= 10:
                synthetic_labels = generate_synthetic_labels_from_telemetry(speed_trace, circuit, gap_to_ahead)
                labels.update(synthetic_labels)
                self.synthetic_labels_created += 1
            
            # Track labeled vs unlabeled
            if labels['overtake_decision'] != -1:
                self.labeled_sequences += 1
            else:
                self.unlabeled_sequences += 1
            
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
            
            # Skip if insufficient telemetry data
            if len(speed_trace) < 10:
                return None
            
            # Calculate Austrian GP specific features
            circuit_similarity = CIRCUIT_SIMILARITY_TO_AUSTRIA.get(circuit, 0.0)
            austrian_relevance = self._calculate_austrian_relevance(speed_trace, circuit)
            high_speed_performance = self._calculate_high_speed_performance(speed_trace)
            drs_effectiveness = self._calculate_drs_effectiveness_score(speed_trace, drs_zones)
            
            # Create enhanced sequence
            sequence = DRSSequence2025Austrian(
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
                overtake_decision=labels['overtake_decision'],
                overtake_success=labels['overtake_success'],
                multi_zone_strategy=labels['multi_zone_strategy'],
                drs_activation=labels['drs_activation'],
                original_label=labels['original_label'],
                label_confidence=labels['label_confidence'],
                label_source=labels['label_source'],
                circuit_similarity_to_austria=circuit_similarity,
                austrian_gp_relevance=austrian_relevance,
                high_speed_sector_performance=high_speed_performance,
                drs_effectiveness_score=drs_effectiveness,
                sequence_length=len(speed_trace)
            )
            
            # Calculate quality score
            sequence.data_quality_score = self._calculate_quality_score(sequence)
            
            return sequence
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"   ❌ Error processing 2025 sequence {index}: {e}")
            self.debug_info['processing_errors'].append(f"Sequence {index}: {str(e)}")
            return None
    
    def _calculate_austrian_relevance(self, speed_trace: List[float], circuit: str) -> float:
        """Calculate how relevant this sequence is for Austrian GP prediction"""
        if not speed_trace:
            return 0.0
        
        speed = np.array(speed_trace)
        max_speed = np.max(speed)
        avg_speed = np.mean(speed)
        
        # Base relevance on circuit similarity
        base_relevance = CIRCUIT_SIMILARITY_TO_AUSTRIA.get(circuit, 0.0)
        
        # Adjust based on speed characteristics similar to Austria
        austria_typical_max = 300.0  # Typical max speed at Austria
        austria_typical_avg = 245.0  # Typical average speed
        
        speed_similarity = 1.0 - abs(max_speed - austria_typical_max) / austria_typical_max
        avg_similarity = 1.0 - abs(avg_speed - austria_typical_avg) / austria_typical_avg
        
        # Combine factors
        relevance = (base_relevance * 0.6 + speed_similarity * 0.2 + avg_similarity * 0.2)
        return max(0.0, min(1.0, relevance))
    
    def _calculate_high_speed_performance(self, speed_trace: List[float]) -> float:
        """Calculate performance in high-speed sections (relevant for Austria)"""
        if not speed_trace:
            return 0.0
        
        speed = np.array(speed_trace)
        high_speed_threshold = 250.0  # km/h
        
        high_speed_points = speed[speed > high_speed_threshold]
        if len(high_speed_points) == 0:
            return 0.0
        
        # Calculate performance metrics for high-speed sections
        high_speed_percentage = len(high_speed_points) / len(speed)
        high_speed_consistency = 1.0 - (np.std(high_speed_points) / np.mean(high_speed_points))
        
        return (high_speed_percentage * 0.6 + high_speed_consistency * 0.4)
    
    def _calculate_drs_effectiveness_score(self, speed_trace: List[float], drs_zones: List[int]) -> float:
        """Calculate DRS effectiveness score for this sequence"""
        if not speed_trace or not drs_zones:
            return 0.0
        
        speed = np.array(speed_trace)
        max_speed = np.max(speed)
        speed_range = np.max(speed) - np.min(speed)
        
        # DRS effectiveness increases with higher max speeds and larger speed ranges
        speed_factor = min(max_speed / 300.0, 1.0)  # Normalize to 300 km/h
        range_factor = min(speed_range / 100.0, 1.0)  # Normalize to 100 km/h range
        zone_factor = min(len(drs_zones) / 3.0, 1.0)  # Austria has 3 DRS zones
        
        return (speed_factor * 0.5 + range_factor * 0.3 + zone_factor * 0.2)
    
    def _calculate_quality_score(self, sequence: DRSSequence2025Austrian) -> float:
        """Calculate quality score with Austrian GP focus"""
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
        
        # Label quality bonus
        if sequence.label_confidence > 0.7:
            score *= 1.15
        elif sequence.label_confidence > 0.5:
            score *= 1.1
        elif sequence.label_confidence > 0.0:
            score *= 1.05
        
        # Austrian relevance bonus
        if sequence.circuit_similarity_to_austria > 0.7:
            score *= 1.1
        elif sequence.circuit_similarity_to_austria > 0.5:
            score *= 1.05
        
        # High-speed performance bonus (important for Austria)
        if sequence.high_speed_sector_performance > 0.7:
            score *= 1.08
        
        return min(1.0, score)
    
    def process_all_2025_sequences_for_austria(self, raw_sequences: List[Dict]) -> List[DRSSequence2025Austrian]:
        """Process all 2025 sequences for Austrian GP prediction"""
        processed_sequences = []
        
        print(f"🔄 Processing {len(raw_sequences)} 2025 sequences for Austrian GP prediction...")
        
        for i, raw_seq in enumerate(raw_sequences):
            if i % 1000 == 0 and i > 0:
                print(f"   Progress: {i}/{len(raw_sequences)} ({i/len(raw_sequences)*100:.1f}%)")
            
            sequence = self.process_2025_sequence_for_austria(raw_seq, i)
            if sequence:
                processed_sequences.append(sequence)
        
        # Update debug info
        self.debug_info['labeled_sequences'] = self.labeled_sequences
        self.debug_info['synthetic_labels_created'] = self.synthetic_labels_created
        
        # Analyze label sources
        label_sources = [seq.label_source for seq in processed_sequences]
        unique_sources = list(set(label_sources))
        self.debug_info['label_sources'] = unique_sources
        
        print(f"✅ Successfully processed {len(processed_sequences)} 2025 sequences")
        print(f"📊 Labeled sequences: {self.labeled_sequences}")
        print(f"📊 Unlabeled sequences: {self.unlabeled_sequences}")
        print(f"🧬 Synthetic labels created: {self.synthetic_labels_created}")
        print(f"🎯 Ready for Austrian GP prediction training")
        
        return processed_sequences
    
    def create_austrian_gp_features_and_metadata(self, sequences: List[DRSSequence2025Austrian]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create feature matrix and metadata optimized for Austrian GP prediction"""
        print(f"🔧 Creating Austrian GP prediction features from {len(sequences)} sequences...")
        
        feature_rows = []
        metadata_rows = []
        
        for sequence in sequences:
            if len(sequence.speed_trace) < 5:
                continue
            
            speed = np.array(sequence.speed_trace)
            
            # Base features
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
                
                # Austrian GP specific features
                'circuit_similarity_to_austria': sequence.circuit_similarity_to_austria,
                'austrian_gp_relevance': sequence.austrian_gp_relevance,
                'high_speed_sector_performance': sequence.high_speed_sector_performance,
                'drs_effectiveness_score': sequence.drs_effectiveness_score,
                
                # Labels
                'overtake_decision': sequence.overtake_decision,
                'overtake_success': sequence.overtake_success,
                'drs_activation': sequence.drs_activation,
                'original_label': sequence.original_label,
                'label_confidence': sequence.label_confidence,
                'label_source': sequence.label_source
            }
            
            # Enhanced telemetry features
            if sequence.throttle_trace:
                throttle = np.array(sequence.throttle_trace)
                features['avg_throttle'] = float(np.mean(throttle))
                features['max_throttle'] = float(np.max(throttle))
                features['throttle_std'] = float(np.std(throttle))
            
            if sequence.brake_trace:
                brake = np.array(sequence.brake_trace)
                features['avg_brake'] = float(np.mean(brake))
                features['max_brake'] = float(np.max(brake))
                features['brake_frequency'] = float(np.sum(np.array(brake) > 0.1) / len(brake))
            
            if sequence.ers_deployment:
                ers = np.array(sequence.ers_deployment)
                features['ers_avg'] = float(np.mean(ers))
                features['ers_max'] = float(np.max(ers))
                features['ers_usage_frequency'] = float(np.sum(ers > 0) / len(ers))
            
            if sequence.gap_to_ahead:
                gap = np.array([g for g in sequence.gap_to_ahead if g > 0])
                if len(gap) > 0:
                    features['avg_gap_to_ahead'] = float(np.mean(gap))
                    features['min_gap_to_ahead'] = float(np.min(gap))
                    features['gap_variation'] = float(np.std(gap))
            
            # Austrian-specific computed features
            features['austria_speed_similarity'] = self._calculate_austria_speed_similarity(speed)
            features['high_speed_percentage'] = float(np.sum(speed > 250) / len(speed))
            features['max_speed_sector'] = self._identify_max_speed_sector(speed)
            
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
                'data_format_version': sequence.data_format_version,
                
                # Austrian GP metadata
                'circuit_similarity_to_austria': sequence.circuit_similarity_to_austria,
                'austrian_gp_relevance': sequence.austrian_gp_relevance,
                'austria_prediction_weight': self._calculate_prediction_weight(sequence),
                
                # Labels
                'overtake_decision': sequence.overtake_decision,
                'overtake_success': sequence.overtake_success,
                'drs_activation': sequence.drs_activation,
                'original_label': sequence.original_label,
                'label_confidence': sequence.label_confidence,
                'label_source': sequence.label_source,
                'is_labeled': sequence.overtake_decision != -1,
                'needs_labeling': sequence.overtake_decision == -1,
                'synthetic_label': 'synthetic' in sequence.label_source,
                'austria_transfer_ready': sequence.circuit_similarity_to_austria > 0.5
            }
            metadata_rows.append(metadata)
        
        feature_df = pd.DataFrame(feature_rows)
        metadata_df = pd.DataFrame(metadata_rows)
        
        print(f"✅ Created Austrian GP prediction feature matrix: {feature_df.shape}")
        
        return feature_df, metadata_df
    
    def _calculate_austria_speed_similarity(self, speed: np.ndarray) -> float:
        """Calculate how similar speed profile is to typical Austria patterns"""
        max_speed = np.max(speed)
        avg_speed = np.mean(speed)
        
        # Austria typical values
        austria_max = 300.0
        austria_avg = 245.0
        
        max_similarity = 1.0 - abs(max_speed - austria_max) / austria_max
        avg_similarity = 1.0 - abs(avg_speed - austria_avg) / austria_avg
        
        return (max_similarity + avg_similarity) / 2.0
    
    def _identify_max_speed_sector(self, speed: np.ndarray) -> int:
        """Identify which sector had the maximum speed (1, 2, or 3)"""
        if len(speed) < 3:
            return 1
        
        sector_size = len(speed) // 3
        sector1_max = np.max(speed[:sector_size])
        sector2_max = np.max(speed[sector_size:2*sector_size])
        sector3_max = np.max(speed[2*sector_size:])
        
        max_speeds = [sector1_max, sector2_max, sector3_max]
        return int(np.argmax(max_speeds) + 1)
    
    def _calculate_prediction_weight(self, sequence: DRSSequence2025Austrian) -> float:
        """Calculate how much weight this sequence should have in Austrian GP prediction"""
        weight = sequence.circuit_similarity_to_austria * 0.4
        weight += sequence.data_quality_score * 0.3
        weight += sequence.label_confidence * 0.2
        weight += sequence.high_speed_sector_performance * 0.1
        
        return weight

def main():
    """Main function for 2025 Austrian GP DRS Decision AI preprocessing"""
    print("🏎️  F1 2025 Austrian GP DRS Decision AI Processor")
    print("=" * 60)
    print("🎯 MISSION: Predict DRS strategies for 2025 Austrian Grand Prix")
    print("📊 TRAINING DATA: 2025 Rounds 1-8 (Bahrain through Canada)")
    print("🏁 TARGET RACE: Austria GP - June 29, 2025 (Round 9)")
    print("=" * 60)
    
    # Check data path
    if not Path(DATA_PATH).exists():
        print(f"❌ Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH variable")
        return
    
    # Initialize processor
    processor = F1_2025_AustrianGP_Processor(DATA_PATH)
    
    # Load 2025 data
    raw_sequences = processor.load_2025_data_for_austria_prediction()
    
    if not raw_sequences:
        print("❌ No 2025 sequences loaded")
        return
    
    # Process sequences for Austrian GP prediction
    sequences = processor.process_all_2025_sequences_for_austria(raw_sequences)
    
    if not sequences:
        print("❌ No 2025 sequences processed successfully")
        return
    
    # Create Austrian GP optimized features
    feature_df, metadata_df = processor.create_austrian_gp_features_and_metadata(sequences)
    
    # Save data
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    feature_df.to_csv(output_path / '2025_austria_features.csv', index=False)
    metadata_df.to_csv(output_path / '2025_austria_metadata.csv', index=False)
    
    with open(output_path / '2025_austria_sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    
    # Save Austrian GP specific data
    debug_info_clean = convert_sets_to_lists(processor.debug_info)
    with open(output_path / '2025_austria_debug_info.json', 'w') as f:
        json.dump(debug_info_clean, f, indent=2)
    
    # Create training/validation split for Austrian GP prediction
    labeled_sequences = feature_df[feature_df['overtake_decision'] != -1]
    if len(labeled_sequences) > 0:
        # Split by circuit similarity (use high similarity circuits for training)
        high_similarity = labeled_sequences[labeled_sequences['circuit_similarity_to_austria'] > 0.7]
        medium_similarity = labeled_sequences[labeled_sequences['circuit_similarity_to_austria'].between(0.4, 0.7)]
        
        # Save training sets
        if len(high_similarity) > 0:
            high_similarity.to_csv(output_path / '2025_austria_training_high_sim.csv', index=False)
        if len(medium_similarity) > 0:
            medium_similarity.to_csv(output_path / '2025_austria_training_medium_sim.csv', index=False)
    
    # Print results
    print("\n" + "="*60)
    print("🏁 F1 2025 AUSTRIAN GP PREDICTION PREPROCESSING COMPLETE!")
    print("="*60)
    
    labeled_count = (metadata_df['overtake_decision'] != -1).sum()
    synthetic_count = metadata_df['synthetic_label'].sum() if 'synthetic_label' in metadata_df.columns else 0
    austria_ready_count = (metadata_df['austria_transfer_ready'] == True).sum()
    
    print(f"\n📊 AUSTRIAN GP PREDICTION DATASET:")
    print(f"   Total sequences: {len(sequences):,}")
    print(f"   Available circuits: {metadata_df['circuit'].nunique()}")
    print(f"   Labeled sequences: {labeled_count:,} ({labeled_count/len(metadata_df)*100:.1f}%)")
    print(f"   Synthetic labels: {synthetic_count:,}")
    print(f"   Austria-ready sequences: {austria_ready_count:,}")
    print(f"   Average quality: {metadata_df['data_quality_score'].mean():.3f}")
    print(f"   Average label confidence: {metadata_df['label_confidence'].mean():.3f}")
    
    # Circuit analysis
    print(f"\n🏁 CIRCUIT ANALYSIS FOR AUSTRIA PREDICTION:")
    circuit_stats = metadata_df.groupby('circuit').agg({
        'sequence_id': 'count',
        'circuit_similarity_to_austria': 'mean',
        'data_quality_score': 'mean',
        'overtake_decision': lambda x: (x != -1).sum()
    }).round(3)
    circuit_stats.columns = ['sequences', 'austria_similarity', 'quality', 'labeled']
    
    for circuit, stats in circuit_stats.iterrows():
        similarity = stats['austria_similarity']
        emoji = "🎯" if similarity > 0.7 else "✅" if similarity > 0.5 else "⚠️"
        print(f"   {emoji} {circuit.upper()}: {stats['sequences']:,} sequences, "
              f"similarity={similarity:.2f}, labeled={stats['labeled']:,}")
    
    # Prediction readiness
    print(f"\n🎯 AUSTRIAN GP PREDICTION READINESS:")
    high_sim_circuits = metadata_df[metadata_df['circuit_similarity_to_austria'] > 0.7]['circuit'].unique()
    if len(high_sim_circuits) > 0:
        print(f"   ✅ High similarity circuits: {', '.join(high_sim_circuits)}")
    
    avg_similarity = metadata_df['circuit_similarity_to_austria'].mean()
    print(f"   📊 Overall Austria similarity: {avg_similarity:.3f}")
    
    if labeled_count > 1000:
        print(f"   ✅ Sufficient labeled data for ML training")
    elif labeled_count > 100:
        print(f"   ⚠️  Limited labeled data - consider synthetic label generation")
    else:
        print(f"   ❌ Insufficient labeled data - enable GENERATE_SYNTHETIC_LABELS")
    
    print(f"\n📁 AUSTRIAN GP PREDICTION FILES CREATED:")
    print(f"   2025_austria_features.csv ({feature_df.shape})")
    print(f"   2025_austria_metadata.csv")
    print(f"   2025_austria_sequences.pkl")
    print(f"   2025_austria_debug_info.json")
    if len(labeled_sequences) > 0:
        print(f"   Training splits by similarity level")
    
    print(f"\n🚀 NEXT STEPS FOR AUSTRIAN GP PREDICTION:")
    print(f"   1. ✅ Load 2025_austria_features.csv for ML training")
    print(f"   2. 🔧 Train DRS decision models on high-similarity circuits")
    print(f"   3. 🎯 Apply transfer learning from 2024 Austrian GP data")
    print(f"   4. 📊 Validate on medium-similarity circuits")
    print(f"   5. 🏁 Deploy for 2025 Austrian GP race strategy")

if __name__ == "__main__":
    main()