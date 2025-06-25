"""
Created on Wed Jun 25 11:17:51 2025

@author: sid

Enhanced F1 AI System - Improved Data Complexity
Making the data more realistic to better test model capabilities

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class RealWorldF1DataGenerator:
    """
    Generate more realistic, complex F1 data that challenges all models
    """
    
    def __init__(self):
        self.noise_factors = {
            'weather_variability': 0.15,
            'tire_degradation_randomness': 0.2,
            'traffic_unpredictability': 0.25,
            'driver_form_variation': 0.1
        }
    
    def generate_complex_race_sequence(self, driver: str, track: str, sequence_length: int) -> Dict:
        """Generate complex, realistic race scenarios"""
        np.random.seed(hash(driver + track + str(time.time())) % 2**31)
        
        # Complex track characteristics
        track_complexity = {
            'sao_paulo': {
                'elevation_changes': 0.8,
                'weather_instability': 0.9,  # High for São Paulo
                'overtaking_zones': 3,
                'drs_effectiveness': 0.7,
                'tire_degradation_rate': 1.2
            },
            'monaco': {
                'elevation_changes': 0.3,
                'weather_instability': 0.1,
                'overtaking_zones': 1,
                'drs_effectiveness': 0.3,
                'tire_degradation_rate': 0.8
            },
            'monza': {
                'elevation_changes': 0.1,
                'weather_instability': 0.3,
                'overtaking_zones': 4,
                'drs_effectiveness': 1.0,
                'tire_degradation_rate': 1.1
            }
        }
        
        track_data = track_complexity.get(track, track_complexity['sao_paulo'])
        
        # Driver complexity (more nuanced than simple aggression)
        driver_characteristics = {
            'verstappen': {
                'raw_speed': 0.98,
                'racecraft': 0.95,
                'consistency': 0.90,
                'pressure_handling': 0.98,
                'tire_management': 0.85,
                'energy_management': 0.90
            },
            'hamilton': {
                'raw_speed': 0.95,
                'racecraft': 0.98,
                'consistency': 0.95,
                'pressure_handling': 0.95,
                'tire_management': 0.95,
                'energy_management': 0.92
            }
        }
        
        driver_data = driver_characteristics.get(driver, driver_characteristics['verstappen'])
        
        sequence = {'driver_id': driver, 'track_id': track, 'telemetry': []}
        
        # Dynamic race state
        race_state = {
            'weather_factor': 1.0,
            'track_temperature': 35.0,
            'tire_compound': np.random.choice(['soft', 'medium', 'hard']),
            'fuel_load': 1.0,
            'ers_deployment_strategy': np.random.choice(['aggressive', 'conservative', 'balanced']),
            'safety_car_probability': 0.02,
            'rain_probability': 0.1 if track == 'sao_paulo' else 0.05
        }
        
        for t in range(sequence_length):
            # Dynamic weather (especially for São Paulo)
            if np.random.random() < race_state['rain_probability']:
                race_state['weather_factor'] *= np.random.uniform(0.8, 1.2)
                race_state['track_temperature'] *= np.random.uniform(0.9, 1.1)
            
            # Complex speed calculation
            base_speed = self._calculate_complex_speed(
                track_data, driver_data, race_state, t, sequence_length
            )
            
            # Realistic throttle with driver skill and conditions
            throttle = self._calculate_realistic_throttle(
                driver_data, race_state, base_speed, t
            )
            
            # Context-aware braking
            brake = self._calculate_context_braking(
                track_data, driver_data, t, sequence_length
            )
            
            # Sophisticated ERS management
            ers = self._calculate_ers_strategy(
                race_state, driver_data, t, sequence_length
            )
            
            # Dynamic position and gaps (complex P17→P1 for Verstappen)
            position, gap_ahead, gap_behind = self._calculate_dynamic_position(
                driver, track, t, sequence_length, driver_data
            )
            
            # Complex DRS availability
            drs_available = self._calculate_drs_availability(
                track_data, position, gap_ahead, t
            )
            
            # Realistic gear calculation
            gear = min(8, max(1, int(base_speed / 45) + np.random.randint(-1, 2)))
            
            telemetry_point = {
                'speed': base_speed,
                'throttle': throttle,
                'brake': brake,
                'ers': ers,
                'gear': gear,
                'gap_ahead': gap_ahead,
                'gap_behind': gap_behind,
                'position': position,
                'drs_available': drs_available,
                'weather_factor': race_state['weather_factor'],
                'track_temp': race_state['track_temperature'],
                'tire_age': min(1.0, (t / sequence_length) * track_data['tire_degradation_rate']),
                'fuel_level': max(0.1, race_state['fuel_load'] - (t / sequence_length) * 0.8),
                'sector': (t % 30) // 10 + 1,
                'lap_progress': (t % 30) / 30.0
            }
            
            sequence['telemetry'].append(telemetry_point)
            
            # Update race state
            race_state['fuel_load'] = telemetry_point['fuel_level']
        
        return sequence
    
    def _calculate_complex_speed(self, track_data, driver_data, race_state, t, seq_len):
        """Calculate speed with multiple complex factors"""
        # Base track speed
        base_speeds = {
            'sao_paulo': 240,
            'monaco': 180,
            'monza': 320,
            'silverstone': 260
        }
        
        base_speed = base_speeds.get('sao_paulo', 240)  # Default to São Paulo
        
        # Driver skill impact
        skill_factor = 1.0 + (driver_data['raw_speed'] - 0.9) * 0.3
        
        # Weather impact
        weather_impact = race_state['weather_factor']
        if weather_impact < 0.9:  # Wet conditions
            base_speed *= 0.7  # Significant speed reduction
        
        # Tire degradation
        tire_age = min(1.0, t / seq_len)
        tire_impact = 1.0 - (tire_age * 0.15)  # Up to 15% speed loss
        
        # Track temperature
        temp_impact = 1.0 + (race_state['track_temperature'] - 35) * 0.001
        
        # Add realistic noise
        noise = np.random.normal(0, 25)
        
        final_speed = base_speed * skill_factor * weather_impact * tire_impact * temp_impact + noise
        
        return np.clip(final_speed, 50, 350)
    
    def _calculate_realistic_throttle(self, driver_data, race_state, speed, t):
        """Calculate throttle with driver skill and conditions"""
        # Base throttle based on speed
        base_throttle = min(1.0, speed / 300.0)
        
        # Driver consistency factor
        consistency_factor = driver_data['consistency']
        throttle_noise = np.random.normal(0, 0.1 * (1 - consistency_factor))
        
        # Weather adaptation
        if race_state['weather_factor'] < 0.9:  # Wet
            base_throttle *= 0.8  # More conservative in wet
        
        # Fuel saving strategy
        if race_state['fuel_load'] < 0.3:  # Low fuel
            base_throttle *= 0.95
        
        throttle = base_throttle + throttle_noise
        return np.clip(throttle, 0, 1)
    
    def _calculate_context_braking(self, track_data, driver_data, t, seq_len):
        """Calculate braking with track and driver context"""
        # Track-specific braking zones
        sector_progress = (t % 30) / 30.0
        
        # São Paulo has heavy braking zones
        if sector_progress > 0.3 and sector_progress < 0.4:  # Sector 1 braking
            base_brake = 0.6
        elif sector_progress > 0.7 and sector_progress < 0.8:  # Sector 2 braking
            base_brake = 0.8
        else:
            base_brake = 0.1
        
        # Driver skill in braking
        skill_factor = driver_data['racecraft']
        brake_variation = np.random.exponential(0.1) * (1 - skill_factor)
        
        brake = base_brake + brake_variation
        return np.clip(brake, 0, 1)
    
    def _calculate_ers_strategy(self, race_state, driver_data, t, seq_len):
        """Calculate ERS with strategic deployment"""
        strategy = race_state['ers_deployment_strategy']
        
        # Base ERS level
        if strategy == 'aggressive':
            base_ers = 0.8
        elif strategy == 'conservative':
            base_ers = 0.4
        else:  # balanced
            base_ers = 0.6
        
        # Driver energy management skill
        management_skill = driver_data['energy_management']
        
        # Lap progress - deploy more in certain zones
        lap_progress = (t % 30) / 30.0
        if lap_progress < 0.3:  # Start/finish straight
            ers_boost = 0.3
        else:
            ers_boost = 0.0
        
        # Add strategic variation
        ers_noise = np.random.normal(0, 0.1 * (1 - management_skill))
        
        ers = base_ers + ers_boost + ers_noise
        return np.clip(ers, 0, 1)
    
    def _calculate_dynamic_position(self, driver, track, t, seq_len, driver_data):
        """Calculate position with realistic P17→P1 progression"""
        
        if driver == 'verstappen' and track == 'sao_paulo':
            # Realistic P17→P1 comeback with challenges
            progress = t / seq_len
            
            # Non-linear progression with realistic obstacles
            if progress < 0.2:  # Early phase: slow progress
                position_gain = progress * 3  # Gain 3 positions
            elif progress < 0.5:  # Mid phase: rapid progress
                position_gain = 3 + (progress - 0.2) * 20  # Gain 6 more positions
            elif progress < 0.8:  # Late phase: fighting for podium
                position_gain = 9 + (progress - 0.5) * 20  # Gain 6 more positions
            else:  # Final phase: P1 battle
                position_gain = 15 + (progress - 0.8) * 5  # Final push to P1
            
            # Add realistic variability
            position_noise = np.random.normal(0, 1)
            position = max(1, 17 - position_gain + position_noise)
            
            # Dynamic gaps based on position
            if position > 10:
                gap_ahead = np.random.exponential(2.0)
                gap_behind = np.random.exponential(1.5)
            elif position > 5:
                gap_ahead = np.random.exponential(1.0)
                gap_behind = np.random.exponential(2.0)
            elif position > 2:
                gap_ahead = np.random.exponential(0.5)
                gap_behind = np.random.exponential(3.0)
            else:  # Leading or P2
                gap_ahead = 999 if position == 1 else np.random.exponential(0.3)
                gap_behind = np.random.exponential(5.0)
                
        else:
            # Regular position dynamics for other drivers
            position = np.random.randint(1, 21)
            gap_ahead = np.random.exponential(1.5)
            gap_behind = np.random.exponential(1.5)
        
        return int(position), max(0.1, gap_ahead), max(0.1, gap_behind)
    
    def _calculate_drs_availability(self, track_data, position, gap_ahead, t):
        """Calculate DRS availability with realistic constraints"""
        # Track has DRS zones
        sector_progress = (t % 30) / 30.0
        
        # São Paulo DRS zones (main straight)
        in_drs_zone = (sector_progress > 0.8 or sector_progress < 0.1)
        
        # Must be within 1 second and not leading
        gap_requirement = gap_ahead < 1.0 and position > 1
        
        # Weather restrictions
        weather_ok = True  # Simplified for now
        
        # Random system failures
        system_ok = np.random.random() > 0.05  # 5% failure rate
        
        return 1 if (in_drs_zone and gap_requirement and weather_ok and system_ok) else 0

def create_complex_strategic_labels(df: pd.DataFrame, driver_id: str = 'verstappen') -> np.ndarray:
    """
    Create complex strategic labels that are harder to predict
    """
    labels = []
    
    for i in range(len(df)):
        row = df.iloc[i] if hasattr(df, 'iloc') else df[i]
        
        # Get multiple factors
        speed = row.get('speed', 0)
        gap_ahead = row.get('gap_ahead', 999)
        position = row.get('position', 10)
        throttle = row.get('throttle', 0)
        ers = row.get('ers', 0)
        tire_age = row.get('tire_age', 0)
        weather_factor = row.get('weather_factor', 1.0)
        fuel_level = row.get('fuel_level', 1.0)
        track_temp = row.get('track_temp', 35)
        
        # Complex decision matrix
        decision_score = 0
        
        # Speed advantage factor
        if speed > 250:
            decision_score += 2
        elif speed > 220:
            decision_score += 1
        
        # Gap opportunity
        if gap_ahead < 0.3:
            decision_score += 3  # Very close
        elif gap_ahead < 0.8:
            decision_score += 2  # Close
        elif gap_ahead < 1.5:
            decision_score += 1  # Moderately close
        
        # Position pressure
        if position > 15:
            decision_score += 2  # Must attack from back
        elif position > 10:
            decision_score += 1
        elif position <= 3:
            decision_score -= 1  # Conservative when in podium
        
        # Resource availability
        if ers > 0.7 and fuel_level > 0.5:
            decision_score += 1
        elif ers < 0.3 or fuel_level < 0.3:
            decision_score -= 2
        
        # Tire condition
        if tire_age > 0.8:
            decision_score -= 2  # Worn tires
        elif tire_age < 0.3:
            decision_score += 1  # Fresh tires
        
        # Weather conditions
        if weather_factor < 0.9:  # Wet/changing conditions
            decision_score -= 1  # More conservative
        
        # Track temperature (affects tire performance)
        if track_temp > 40:
            decision_score -= 1  # Hot track, conservative
        
        # Driver-specific adjustments
        if driver_id == 'verstappen':
            decision_score += 1  # More aggressive
        elif driver_id == 'hamilton':
            if position <= 3:
                decision_score -= 1  # More strategic when ahead
        
        # Add some randomness for realism
        random_factor = np.random.uniform(-1, 1)
        decision_score += random_factor
        
        # Convert to strategic decision
        if decision_score >= 4:
            decision = 1  # Attempt Overtake
        elif decision_score >= 1:
            decision = 2  # Apply Pressure
        else:
            decision = 0  # Hold Position
        
        labels.append(decision)
    
    return np.array(labels)

def run_enhanced_benchmark():
    """
    Run benchmark with more complex, realistic data
    """
    print("🏁 ENHANCED F1 AI BENCHMARK - REALISTIC DATA EDITION")
    print("🇳🇱 Testing with Complex P17→P1 São Paulo GP Scenarios")
    print("=" * 70)
    
    # Generate more complex data
    data_generator = RealWorldF1DataGenerator()
    
    print("📊 Generating complex race scenarios...")
    race_data = []
    
    # Generate diverse scenarios
    drivers = ['verstappen', 'hamilton', 'leclerc', 'russell']
    tracks = ['sao_paulo', 'monaco', 'monza', 'silverstone']
    
    for i in range(100):
        driver = np.random.choice(drivers)
        track = np.random.choice(tracks)
        
        # Bias toward Verstappen + São Paulo for P17→P1 scenarios
        if i % 4 == 0:
            driver = 'verstappen'
            track = 'sao_paulo'
        
        sequence = data_generator.generate_complex_race_sequence(
            driver=driver,
            track=track,
            sequence_length=np.random.randint(40, 80)
        )
        race_data.append(sequence)
    
    print(f"✅ Generated {len(race_data)} complex race scenarios")
    
    # Prepare features for traditional models
    print("\n🔧 Preparing traditional ML features...")
    from sklearn.preprocessing import StandardScaler
    
    all_features = []
    all_labels = []
    
    for race in race_data:
        df = pd.DataFrame(race['telemetry'])
        driver_id = race['driver_id']
        
        # Extract features (more comprehensive)
        features = []
        for _, row in df.iterrows():
            feature_vector = [
                row['speed'] / 350.0,  # Normalized speed
                row['throttle'],
                row['brake'],
                row['ers'],
                row['gear'] / 8.0,
                row['gap_ahead'],
                row['gap_behind'],
                row['position'] / 20.0,
                row['weather_factor'],
                row['tire_age'],
                row['fuel_level'],
                row['track_temp'] / 50.0,
                row['lap_progress'],
                row['drs_available']
            ]
            features.append(feature_vector)
        
        # Create complex labels
        labels = create_complex_strategic_labels(df, driver_id)
        
        # Sample every 3rd point to reduce correlation
        for j in range(0, len(features), 3):
            all_features.append(features[j])
            all_labels.append(labels[j])
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Label distribution: {np.bincount(y)}")
    
    # Train models
    print("\n🚀 Training models on complex data...")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Random Forest
    print("   Training Random Forest...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_time = time.time() - start_time
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    
    print(f"✅ Random Forest Results:")
    print(f"   📊 Accuracy: {rf_accuracy:.3f}")
    print(f"   📊 F1-Score: {rf_f1:.3f}")
    print(f"   ⏱️  Training Time: {rf_time:.3f}s")
    
    # Feature importance analysis
    feature_names = [
        'Speed', 'Throttle', 'Brake', 'ERS', 'Gear', 'Gap Ahead', 'Gap Behind',
        'Position', 'Weather', 'Tire Age', 'Fuel Level', 'Track Temp', 'Lap Progress', 'DRS'
    ]
    
    importance = rf_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    
    print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.3f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Feature importance
    plt.subplot(1, 3, 1)
    plt.bar(range(5), importance[sorted_idx[:5]], color='lightblue', alpha=0.7)
    plt.title('Top 5 Feature Importance')
    plt.xticks(range(5), [feature_names[sorted_idx[i]] for i in range(5)], rotation=45)
    plt.ylabel('Importance')
    
    # Label distribution
    plt.subplot(1, 3, 2)
    strategy_names = ['Hold Position', 'Attempt Overtake', 'Apply Pressure']
    label_counts = np.bincount(y)
    plt.pie(label_counts, labels=strategy_names, autopct='%1.1f%%', 
            colors=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Strategic Decision Distribution')
    
    # Performance comparison
    plt.subplot(1, 3, 3)
    metrics = ['Accuracy', 'F1-Score']
    simple_scores = [1.0, 1.0]  # From previous simple data
    complex_scores = [rf_accuracy, rf_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, simple_scores, width, label='Simple Data', color='lightgray', alpha=0.7)
    plt.bar(x + width/2, complex_scores, width, label='Complex Data', color='orange', alpha=0.7)
    plt.ylabel('Score')
    plt.title('Simple vs Complex Data')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1.1)
    
    plt.suptitle('Enhanced F1 AI Analysis - Complex Data Results', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎯 ANALYSIS:")
    if rf_accuracy < 0.9:
        print(f"✅ SUCCESS: Complex data is challenging models appropriately!")
        print(f"🧠 The AI must now learn complex racing patterns")
        print(f"🏁 More realistic for real F1 deployment")
    else:
        print(f"⚠️  Data might still be too simple")
    
    print(f"\n🇳🇱 VERSTAPPEN P17→P1 INSIGHTS:")
    print(f"🏆 Complex comeback scenarios included")
    print(f"🎯 Strategic pressure modeling active")
    print(f"⚡ Weather and tire complexity added")
    print(f"🚀 Ready for advanced F1 AI development!")

if __name__ == "__main__":
    run_enhanced_benchmark()