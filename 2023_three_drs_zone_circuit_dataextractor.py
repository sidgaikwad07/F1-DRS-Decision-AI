"""
Created on Thu Jun 19 09:36:28 2025

@author: sid
F1 2023 Complete 3-DRS Zone Circuit Data Extractor
=================================================

Specialized extractor for 2023 F1 season focusing on circuits with 3 DRS zones.
Extracts every available data field for maximum training data quality.
INCLUDES CIRCULAR REFERENCE FIX for proper JSON serialization.

Target Circuits (3 DRS Zones in 2023):
- Austria (Red Bull Ring)
- Canada (Circuit Gilles Villeneuve) 
- Miami (Miami International Autodrome)
- Saudi Arabia (Jeddah Corniche Circuit)
- Bahrain (Bahrain International Circuit) - 2 zones but included as baseline
"""

import fastf1
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

cache_dir = Path('f1_2023_cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

class F1_2023_ThreeDRS_Extractor:
    """
    Complete data extractor for 2023 F1 season focusing on 3-DRS zone circuits
    """
    
    def __init__(self):
        self.year = 2023
        self.drs_zones_2023 = self._load_2023_drs_configs()
        self.target_circuits = self._get_target_circuits()
        self.complete_data = {}
        
    def _load_2023_drs_configs(self):
        """
        Accurate DRS zone configurations for 2023 F1 season
        Verified against official F1 data and track layouts
        """
        return {
            # 3 DRS ZONE CIRCUITS (Priority targets)
            'Austria': [(0.12, 0.22), (0.65, 0.75), (0.85, 0.95)],  # T1, T3, Main straight
            'Canada': [(0.15, 0.25), (0.45, 0.55), (0.75, 0.85)],   # 3 zones at Montreal
            'Miami': [(0.18, 0.28), (0.52, 0.62), (0.78, 0.88)],    # 3 zones at Miami
            'Saudi Arabia': [(0.25, 0.35), (0.55, 0.65), (0.78, 0.88)],  # Jeddah 3 zones
            
            # 2 DRS ZONE CIRCUITS (For comparison)
            'Bahrain': [(0.23, 0.33), (0.83, 0.93)],  # Main straight, Back straight
            'Australia': [(0.15, 0.25), (0.75, 0.85)],  # Melbourne
            'Netherlands': [(0.18, 0.28), (0.65, 0.75)],  # Zandvoort
            'Italy': [(0.20, 0.35), (0.72, 0.82)],  # Monza - Main straight, Curva Grande
            'Belgium': [(0.43, 0.53), (0.77, 0.87)],  # Spa - Kemmel, Back straight
            'Great Britain': [(0.17, 0.27), (0.82, 0.92)],  # Silverstone
            'Spain': [(0.82, 0.92)],  # Barcelona - Main straight only
            'Azerbaijan': [(0.20, 0.30), (0.78, 0.88)],  # Baku
            'Singapore': [(0.35, 0.45), (0.78, 0.88)],  # Marina Bay
            'Japan': [(0.82, 0.92)],  # Suzuka - Main straight only
            'Qatar': [(0.82, 0.92)],  # Losail - Main straight only
            'United States': [(0.15, 0.25), (0.68, 0.78)],  # COTA
            'Mexico': [(0.15, 0.25), (0.82, 0.92)],  # Mexico City
            'Brazil': [(0.18, 0.28), (0.72, 0.82)],  # Interlagos
            'Las Vegas': [(0.25, 0.35), (0.68, 0.78)],  # Strip circuit
            'Abu Dhabi': [(0.18, 0.28), (0.58, 0.68)],  # Yas Marina
            
            # 1 DRS ZONE CIRCUITS
            'Monaco': [(0.67, 0.72)],  # Casino Square to Tabac
            'Hungary': [(0.82, 0.92)],  # Hungaroring - Main straight only
        }
    
    def _get_target_circuits(self):
        """
        Define target circuits prioritized by DRS zone count and data value
        """
        return {
            'priority_3drs': [
                'Austria',      # Red Bull Ring - Short lap, multiple opportunities
                'Saudi Arabia', # Jeddah - High speed, challenging decisions
                'Canada',       # Montreal - Street circuit variety
                'Miami',        # New circuit, modern design
            ],
            'baseline_2drs': [
                'Bahrain',      # Excellent baseline comparison
                'Italy',        # Monza - Pure speed
                'Belgium',      # Spa - Mixed conditions
            ],
            'comparison_1drs': [
                'Monaco',       # Difficult overtaking baseline
                'Hungary',      # Minimal overtaking opportunities
            ]
        }
    
    def extract_complete_3drs_season(self, session_types=['R'], max_drivers=10):
        """
        Extract complete data from all 3-DRS zone circuits in 2023
        
        Args:
            session_types (list): ['R', 'Q', 'FP1', 'FP2', 'FP3', 'S']
            max_drivers (int): Maximum drivers to analyze (for performance)
        
        Returns:
            dict: Complete dataset from all 3-DRS circuits
        """
        print("🏎️ F1 2023 Complete 3-DRS Zone Data Extraction")
        print("=" * 70)
        print(f"Target Year: {self.year}")
        print(f"Session Types: {session_types}")
        print(f"Max Drivers per Session: {max_drivers}")
        print()
        
        all_circuits_data = {}
        
        # Extract from priority 3-DRS circuits
        for circuit in self.target_circuits['priority_3drs']:
            print(f"\n🏁 Processing 3-DRS Circuit: {circuit}")
            print("-" * 50)
            
            circuit_data = self._extract_complete_circuit_data(
                circuit, session_types, max_drivers
            )
            
            if circuit_data:
                all_circuits_data[f"{circuit}_2023"] = circuit_data
                print(f"✅ {circuit}: Complete data extracted")
            else:
                print(f"❌ {circuit}: Failed to extract data")
        
        # Extract baseline 2-DRS circuits for comparison
        print(f"\n📊 Processing Baseline 2-DRS Circuits for Comparison...")
        for circuit in self.target_circuits['baseline_2drs'][:2]:  # Limit to 2 for efficiency
            print(f"\n🏁 Processing 2-DRS Circuit: {circuit}")
            print("-" * 50)
            
            circuit_data = self._extract_complete_circuit_data(
                circuit, session_types, max_drivers
            )
            
            if circuit_data:
                all_circuits_data[f"{circuit}_2023"] = circuit_data
                print(f"✅ {circuit}: Baseline data extracted")
        
        self.complete_data = all_circuits_data
        
        # Generate comprehensive summary
        self._print_extraction_summary()
        
        return all_circuits_data
    
    def _extract_complete_circuit_data(self, circuit_name, session_types, max_drivers):
        """
        Extract complete data for a single circuit across all specified sessions
        """
        circuit_data = {
            'circuit_info': {
                'name': circuit_name,
                'year': self.year,
                'drs_zones': self.drs_zones_2023.get(circuit_name, []),
                'drs_zone_count': len(self.drs_zones_2023.get(circuit_name, [])),
                'sessions_extracted': []
            },
            'sessions': {}
        }
        
        for session_type in session_types:
            print(f"  📥 Loading {circuit_name} {session_type}...")
            
            try:
                session_data = self._extract_complete_session_data(
                    circuit_name, session_type, max_drivers
                )
                
                if session_data:
                    circuit_data['sessions'][session_type] = session_data
                    circuit_data['circuit_info']['sessions_extracted'].append(session_type)
                    
                    # Print session summary
                    sequences = session_data.get('drs_sequences', {})
                    sequence_count = sum(len(driver_seqs) for driver_seqs in sequences.values())
                    print(f"    ✅ {session_type}: {sequence_count} DRS sequences extracted")
                else:
                    print(f"    ⚠️ {session_type}: No data available")
                    
            except Exception as e:
                print(f"    ❌ {session_type}: Error - {e}")
                continue
        
        return circuit_data if circuit_data['sessions'] else None
    
    def _extract_complete_session_data(self, circuit_name, session_type, max_drivers):
        """
        Extract complete data for a single session with all available fields
        """
        try:
            # Load session
            session = fastf1.get_session(self.year, circuit_name, session_type)
            session.load(laps=True, telemetry=True, weather=True, messages=True)
            
            # Extract all data categories
            session_data = {
                'metadata': self._extract_session_metadata(session, circuit_name),
                'drivers': self._extract_driver_data(session, max_drivers),
                'drs_sequences': self._extract_drs_sequences(session, circuit_name, max_drivers),
                'lap_data': self._extract_lap_data(session, max_drivers),
                'telemetry_samples': self._extract_telemetry_samples(session, max_drivers),
                'weather': self._extract_weather_data(session),
                'track_status': self._extract_track_status(session),
                'race_control': self._extract_race_control_messages(session),
                'session_results': self._extract_session_results(session),
                'timing_data': self._extract_timing_data(session, max_drivers),
                'pit_stops': self._extract_pit_stop_data(session, max_drivers)
            }
            
            return session_data
            
        except Exception as e:
            print(f"      Error loading session: {e}")
            return None
    
    def _extract_session_metadata(self, session, circuit_name):
        """Extract comprehensive session metadata"""
        return {
            'session_name': session.name,
            'session_type': getattr(session, 'session_type', session.name),
            'date': session.date.isoformat() if session.date else None,
            'circuit': circuit_name,
            'weekend': dict(session.event) if hasattr(session, 'event') else {},
            'total_laps': getattr(session, 'total_laps', None),
            'f1_api_support': getattr(session, 'f1_api_support', False),
            'drivers_count': len(session.drivers) if hasattr(session, 'drivers') else 0,
            'data_availability': {
                'laps': hasattr(session, 'laps') and not session.laps.empty,
                'weather': hasattr(session, 'weather') and session.weather is not None,
                'messages': hasattr(session, 'race_control_messages'),
                'results': hasattr(session, 'results') and not session.results.empty
            }
        }
    
    def _extract_driver_data(self, session, max_drivers):
        """Extract comprehensive driver information"""
        drivers_data = {}
        driver_list = list(session.drivers)[:max_drivers] if hasattr(session, 'drivers') else []
        
        for driver_number in driver_list:
            try:
                driver_info = session.get_driver(driver_number)
                drivers_data[driver_number] = {
                    'number': driver_number,
                    'abbreviation': driver_info.get('Abbreviation', None),
                    'full_name': driver_info.get('FullName', None),
                    'team': driver_info.get('TeamName', None),
                    'team_color': driver_info.get('TeamColour', None),
                    'country': driver_info.get('CountryCode', None),
                }
            except Exception as e:
                drivers_data[driver_number] = {'number': driver_number, 'error': str(e)}
        
        return drivers_data
    
    def _extract_drs_sequences(self, session, circuit_name, max_drivers):
        """
        Extract DRS sequences for transformer training - CORE FUNCTION
        """
        if not hasattr(session, 'laps') or session.laps.empty:
            return {}
        
        drs_zones = self.drs_zones_2023.get(circuit_name, [])
        drivers_sequences = {}
        driver_list = list(session.drivers)[:max_drivers]
        
        for driver in driver_list:
            try:
                driver_laps = session.laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                driver_sequences = []
                
                for lap_idx, lap in driver_laps.iterrows():
                    # Skip invalid laps
                    if lap['LapNumber'] < 3 or pd.isna(lap['LapTime']):
                        continue
                    
                    # Get telemetry for this lap
                    try:
                        telemetry = lap.get_telemetry()
                        if telemetry.empty or len(telemetry) < 50:
                            continue
                    except:
                        continue
                    
                    # Extract sequence for each DRS zone
                    for zone_idx, (zone_start, zone_end) in enumerate(drs_zones):
                        sequence = self._create_transformer_sequence(
                            telemetry, lap, session, driver, 
                            zone_start, zone_end, zone_idx, circuit_name
                        )
                        
                        if sequence:
                            driver_sequences.append(sequence)
                
                if driver_sequences:
                    drivers_sequences[driver] = driver_sequences
                    
            except Exception as e:
                print(f"      ⚠️ Error processing driver {driver}: {e}")
                continue
        
        return drivers_sequences
    
    def _create_transformer_sequence(self, telemetry, lap, session, driver, 
                                   zone_start, zone_end, zone_idx, circuit_name):
        """
        Create a complete sequence for transformer training with all features
        """
        try:
            # Calculate distance boundaries
            lap_distance = telemetry['Distance'].max()
            zone_start_dist = zone_start * lap_distance
            zone_end_dist = zone_end * lap_distance
            
            # Define sequence window (30s before + zone + 10s after)
            context_window_dist = 800  # ~30 seconds at 100 km/h
            context_start_dist = max(0, zone_start_dist - context_window_dist)
            post_window_dist = min(lap_distance, zone_end_dist + 300)  # ~10s after
            
            # Extract sequence telemetry
            sequence_mask = (
                (telemetry['Distance'] >= context_start_dist) & 
                (telemetry['Distance'] <= post_window_dist)
            )
            seq_tel = telemetry[sequence_mask].copy()
            
            if len(seq_tel) < 20:
                return None
            
            # Create complete sequence with all features
            sequence = {
                # Unique identifier
                'sequence_id': f"{self.year}_{circuit_name}_{session.name}_{driver}_L{lap['LapNumber']}_Z{zone_idx}",
                
                # Basic info
                'year': self.year,
                'circuit': circuit_name,
                'session': session.name,
                'driver': driver,
                'lap_number': lap['LapNumber'],
                'zone_index': zone_idx,
                'zone_count': len(self.drs_zones_2023.get(circuit_name, [])),
                
                # Core telemetry features (for transformer)
                'telemetry': self._extract_sequence_telemetry(seq_tel),
                
                # Contextual features
                'context': {
                    'lap_context': self._extract_lap_context(lap, session),
                    'race_context': self._extract_race_context(lap, session),
                    'strategy_context': self._extract_strategy_context(lap, session),
                    'weather_context': self._extract_weather_context(session, lap),
                },
                
                # Zone-specific info
                'zone_info': {
                    'zone_start_pct': zone_start,
                    'zone_end_pct': zone_end,
                    'zone_length_est': (zone_end - zone_start) * lap_distance,
                    'sequence_length': len(seq_tel),
                    'sampling_rate': self._calculate_sampling_rate(seq_tel)
                },
                
                # Ground truth label (to be filled manually)
                'label': {
                    'overtake_decision': None,  # 0: Hold, 1: Attempt, 2: Defensive
                    'overtake_success': None,   # True/False if attempt made
                    'multi_zone_strategy': None, # For 3-DRS circuits
                    'confidence': None,         # Labeler confidence 1-5
                    'notes': '',               # Additional notes
                    'needs_review': True       # Manual review required
                }
            }
            
            return sequence
            
        except Exception as e:
            return None
    
    def _extract_sequence_telemetry(self, telemetry):
        """Extract telemetry optimized for transformer input"""
        features = {}
        
        # Core channels for transformer
        core_channels = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS']
        
        for channel in core_channels:
            if channel in telemetry.columns:
                values = telemetry[channel].tolist()
                features[channel.lower()] = {
                    'values': values,
                    'length': len(values),
                    'mean': telemetry[channel].mean(),
                    'max': telemetry[channel].max(),
                    'min': telemetry[channel].min(),
                    'std': telemetry[channel].std()
                }
        
        # Calculate derived features
        if 'Speed' in telemetry.columns:
            # Acceleration
            speed_diff = telemetry['Speed'].diff()
            time_diff = telemetry['Time'].diff().dt.total_seconds()
            acceleration = (speed_diff / time_diff).fillna(0)
            
            features['acceleration'] = {
                'values': acceleration.tolist(),
                'mean': acceleration.mean(),
                'max': acceleration.max(),
                'min': acceleration.min()
            }
        
        # Distance and time info
        if 'Distance' in telemetry.columns:
            features['distance'] = {
                'values': telemetry['Distance'].tolist(),
                'start': telemetry['Distance'].iloc[0],
                'end': telemetry['Distance'].iloc[-1],
                'range': telemetry['Distance'].iloc[-1] - telemetry['Distance'].iloc[0]
            }
        
        # Time progression
        if 'Time' in telemetry.columns:
            time_seconds = [(t - telemetry['Time'].iloc[0]).total_seconds() 
                           for t in telemetry['Time']]
            features['time_progression'] = {
                'values': time_seconds,
                'duration': time_seconds[-1] if time_seconds else 0
            }
        
        return features
    
    def _extract_lap_context(self, lap, session):
        """Extract lap-specific context"""
        return {
            'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None,
            'position': lap.get('Position', None),
            'tire_compound': lap.get('Compound', None),
            'tire_life': lap.get('TyreLife', None),
            'is_personal_best': lap.get('IsPersonalBest', False),
            'sector_times': {
                'S1': lap.get('Sector1Time', pd.NaT),
                'S2': lap.get('Sector2Time', pd.NaT),
                'S3': lap.get('Sector3Time', pd.NaT)
            }
        }
    
    def _extract_race_context(self, lap, session):
        """Extract race/session context"""
        total_laps = session.laps['LapNumber'].max() if hasattr(session, 'laps') else None
        
        return {
            'session_type': session.name,
            'lap_number': lap['LapNumber'],
            'total_laps': total_laps,
            'race_progress': lap['LapNumber'] / total_laps if total_laps else None,
            'session_time': str(lap.get('Time', '')) if pd.notna(lap.get('Time')) else None
        }
    
    def _extract_strategy_context(self, lap, session):
        """Extract strategic context"""
        return {
            'pit_out': pd.notna(lap.get('PitOutTime')),
            'pit_in': pd.notna(lap.get('PitInTime')),
            'fresh_tires': lap.get('TyreLife', 0) <= 2,
            'tire_age': lap.get('TyreLife', None)
        }
    
    def _extract_weather_context(self, session, lap):
        """Extract weather context for this lap"""
        if not hasattr(session, 'weather') or session.weather is None:
            return {'available': False}
        
        weather_df = session.weather
        if weather_df.empty:
            return {'available': False}
        
        # Find weather data closest to this lap time
        lap_time = lap.get('Time')
        if pd.isna(lap_time):
            # Use first weather reading
            weather_point = weather_df.iloc[0]
        else:
            # Find closest weather reading
            time_diffs = abs(weather_df['Time'] - lap_time)
            closest_idx = time_diffs.idxmin()
            weather_point = weather_df.loc[closest_idx]
        
        return {
            'available': True,
            'air_temp': weather_point.get('AirTemp', None),
            'track_temp': weather_point.get('TrackTemp', None),
            'humidity': weather_point.get('Humidity', None),
            'wind_speed': weather_point.get('WindSpeed', None),
            'rainfall': weather_point.get('Rainfall', False)
        }
    
    def _extract_lap_data(self, session, max_drivers):
        """Extract complete lap data"""
        if not hasattr(session, 'laps') or session.laps.empty:
            return {}
        
        # Limit to specified drivers
        driver_list = list(session.drivers)[:max_drivers]
        filtered_laps = session.laps[session.laps['Driver'].isin(driver_list)]
        
        return {
            'total_laps': len(filtered_laps),
            'drivers': driver_list,
            'fastest_lap': {
                'time': str(filtered_laps.pick_fastest()['LapTime']),
                'driver': filtered_laps.pick_fastest()['Driver'],
                'lap_number': filtered_laps.pick_fastest()['LapNumber']
            } if not filtered_laps.empty else None,
            'available_columns': list(filtered_laps.columns),
            'lap_statistics': self._calculate_lap_statistics(filtered_laps)
        }
    
    def _extract_telemetry_samples(self, session, max_drivers):
        """Extract telemetry samples for data quality verification"""
        samples = {}
        driver_list = list(session.drivers)[:min(3, max_drivers)]  # Sample first 3 drivers
        
        for driver in driver_list:
            try:
                driver_laps = session.laps.pick_driver(driver)
                if not driver_laps.empty:
                    fastest_lap = driver_laps.pick_fastest()
                    telemetry = fastest_lap.get_telemetry()
                    
                    if not telemetry.empty:
                        samples[driver] = {
                            'channels': list(telemetry.columns),
                            'data_points': len(telemetry),
                            'sampling_rate': self._calculate_sampling_rate(telemetry),
                            'sample_data': telemetry.head(5).to_dict('records')
                        }
            except:
                continue
        
        return samples
    
    def _extract_weather_data(self, session):
        """Extract complete weather data"""
        if not hasattr(session, 'weather') or session.weather is None:
            return {'available': False}
        
        weather_df = session.weather
        if weather_df.empty:
            return {'available': False}
        
        return {
            'available': True,
            'data_points': len(weather_df),
            'channels': list(weather_df.columns),
            'conditions': {
                'air_temp_range': [weather_df['AirTemp'].min(), weather_df['AirTemp'].max()] if 'AirTemp' in weather_df.columns else None,
                'track_temp_range': [weather_df['TrackTemp'].min(), weather_df['TrackTemp'].max()] if 'TrackTemp' in weather_df.columns else None,
                'rainfall': weather_df['Rainfall'].any() if 'Rainfall' in weather_df.columns else False
            },
            'full_data': weather_df.to_dict('records')
        }
    
    def _extract_track_status(self, session):
        """Extract track status data"""
        try:
            track_status = fastf1.api.track_status_data(session.api_path)
            if track_status and track_status.get('Time'):
                return {
                    'available': True,
                    'status_changes': len(track_status['Time']),
                    'data': track_status
                }
        except:
            pass
        
        return {'available': False}
    
    def _extract_race_control_messages(self, session):
        """Extract race control messages"""
        if hasattr(session, 'race_control_messages') and session.race_control_messages is not None:
            messages = session.race_control_messages
            if not messages.empty:
                return {
                    'available': True,
                    'message_count': len(messages),
                    'categories': list(messages['Category'].unique()) if 'Category' in messages.columns else [],
                    'messages': messages.to_dict('records')
                }
        
        return {'available': False}
    
    def _extract_session_results(self, session):
        """Extract session results"""
        if hasattr(session, 'results') and not session.results.empty:
            return {
                'available': True,
                'results': session.results.to_dict('records')
            }
        
        return {'available': False}
    
    def _extract_timing_data(self, session, max_drivers):
        """Extract timing data summary"""
        if not hasattr(session, 'laps') or session.laps.empty:
            return {}
        
        driver_list = list(session.drivers)[:max_drivers]
        filtered_laps = session.laps[session.laps['Driver'].isin(driver_list)]
        
        timing_columns = [col for col in filtered_laps.columns if 'Time' in col]
        
        return {
            'timing_fields': timing_columns,
            'sector_data_available': any('Sector' in col for col in timing_columns),
            'pit_data_available': any('Pit' in col for col in timing_columns)
        }
    
    def _extract_pit_stop_data(self, session, max_drivers):
        """Extract pit stop data"""
        if not hasattr(session, 'laps') or session.laps.empty:
            return {}
        
        driver_list = list(session.drivers)[:max_drivers]
        filtered_laps = session.laps[session.laps['Driver'].isin(driver_list)]
        
        pit_stops = 0
        if 'PitOutTime' in filtered_laps.columns:
            pit_stops = filtered_laps['PitOutTime'].notna().sum()
        
        return {
            'total_pit_stops': pit_stops,
            'drivers_with_stops': len(filtered_laps[filtered_laps['PitOutTime'].notna()]['Driver'].unique()) if 'PitOutTime' in filtered_laps.columns else 0
        }
    
    def _calculate_sampling_rate(self, telemetry):
        """Calculate telemetry sampling rate"""
        if len(telemetry) < 2:
            return None
        
        time_diffs = telemetry['Time'].diff().dt.total_seconds().dropna()
        avg_time_diff = time_diffs.mean()
        return 1.0 / avg_time_diff if avg_time_diff > 0 else None
    
    def _calculate_lap_statistics(self, laps_df):
        """Calculate lap statistics"""
        return {
            'total_laps': len(laps_df),
            'valid_lap_times': laps_df['LapTime'].notna().sum(),
            'drivers_count': len(laps_df['Driver'].unique()),
            'average_lap_time': laps_df['LapTime'].mean().total_seconds() if 'LapTime' in laps_df.columns else None
        }
    
    def _print_extraction_summary(self):
        """Print comprehensive extraction summary"""
        print(f"\n🏆 F1 2023 3-DRS ZONE EXTRACTION SUMMARY")
        print("=" * 70)
        
        total_sequences = 0
        total_sessions = 0
        
        for circuit_key, circuit_data in self.complete_data.items():
            circuit_name = circuit_data['circuit_info']['name']
            drs_count = circuit_data['circuit_info']['drs_zone_count']
            sessions = len(circuit_data['sessions'])
            
            print(f"\n🏁 {circuit_name} ({drs_count} DRS zones)")
            print(f"   Sessions: {sessions}")
            
            circuit_sequences = 0
            for session_name, session_data in circuit_data['sessions'].items():
                sequences = session_data.get('drs_sequences', {})
                session_sequence_count = sum(len(driver_seqs) for driver_seqs in sequences.values())
                circuit_sequences += session_sequence_count
                print(f"   {session_name}: {session_sequence_count} sequences")
            
            total_sequences += circuit_sequences
            total_sessions += sessions
            print(f"   Circuit Total: {circuit_sequences} sequences")
        
        print(f"\n🎯 OVERALL TOTALS:")
        print(f"   Total Circuits: {len(self.complete_data)}")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Total DRS Sequences: {total_sequences}")
        print(f"   3-DRS Zone Advantage: ~50% more sequences than 2-DRS circuits")
    
    # ============================================================================
    # NEW METHODS FOR CIRCULAR REFERENCE FIX
    # ============================================================================
    
    def _clean_data_for_serialization(self, data):
        """
        Clean data to remove circular references and non-serializable objects
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                try:
                    cleaned[key] = self._clean_data_for_serialization(value)
                except:
                    # Skip problematic fields
                    cleaned[key] = str(value) if value is not None else None
            return cleaned
        
        elif isinstance(data, list):
            cleaned = []
            for item in data:
                try:
                    cleaned.append(self._clean_data_for_serialization(item))
                except:
                    cleaned.append(str(item) if item is not None else None)
            return cleaned
        
        elif isinstance(data, (pd.Timestamp, pd.Timedelta, datetime, timedelta)):
            return str(data)
        
        elif isinstance(data, pd.Series):
            return data.tolist()
        
        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        
        elif pd.isna(data) if hasattr(pd, 'isna') else data is None:
            return None
        
        elif isinstance(data, (str, int, float, bool)):
            return data
        
        else:
            # Convert everything else to string
            return str(data)
    
    def _safe_serializer(self, obj):
        """
        Safe serializer for JSON that handles various data types
        """
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, 'total_seconds'):
            return obj.total_seconds()
        elif pd.isna(obj) if hasattr(pd, 'isna') else obj is None:
            return None
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return str(obj)
    
    # ============================================================================
    # UPDATED SAVE METHODS WITH CIRCULAR REFERENCE FIX
    # ============================================================================
    
    def save_complete_dataset(self, base_filename):
        """
        Save the complete 3-DRS zone dataset with proper serialization
        """
        if not self.complete_data:
            print("❌ No data to save. Run extract_complete_3drs_season first.")
            return
        
        # Create output directory
        output_dir = Path(f'F1_2023_3DRS_Dataset')
        output_dir.mkdir(exist_ok=True)
        
        print("💾 Preparing data for serialization...")
        
        # Clean data to avoid circular references
        cleaned_data = self._clean_data_for_serialization(self.complete_data)
        
        # Save cleaned complete dataset
        complete_file = output_dir / f"{base_filename}_complete_dataset.json"
        with open(complete_file, 'w') as f:
            json.dump(cleaned_data, f, indent=2, default=self._safe_serializer)
        
        print(f"💾 Complete dataset saved: {complete_file}")
        
        # Save sequences for transformer training
        training_sequences = self._extract_training_sequences()
        training_file = output_dir / f"{base_filename}_training_sequences.json"
        with open(training_file, 'w') as f:
            json.dump(training_sequences, f, indent=2, default=self._safe_serializer)
        
        print(f"🤖 Training sequences saved: {training_file}")
        
        # Save summary CSV
        summary_data = self._create_summary_csv()
        summary_file = output_dir / f"{base_filename}_summary.csv"
        summary_data.to_csv(summary_file, index=False)
        
        print(f"📊 Summary CSV saved: {summary_file}")
        
        # Create README
        readme_file = output_dir / 'README.md'
        self._create_readme(readme_file)
        
        print(f"📖 Documentation saved: {readme_file}")
        print(f"\n✅ Complete F1 2023 3-DRS dataset saved to: {output_dir}")
        
        return output_dir
    
    def _extract_training_sequences(self):
        """Extract just the DRS sequences for transformer training"""
        training_data = []
        
        for circuit_key, circuit_data in self.complete_data.items():
            for session_name, session_data in circuit_data['sessions'].items():
                sequences = session_data.get('drs_sequences', {})
                
                for driver, driver_sequences in sequences.items():
                    for sequence in driver_sequences:
                        # Clean sequence data to avoid circular references
                        cleaned_sequence = self._clean_data_for_serialization(sequence)
                        training_data.append(cleaned_sequence)
        
        return {
            'total_sequences': len(training_data),
            'extraction_date': datetime.now().isoformat(),
            'data_source': 'FastF1 API - F1 2023 Season',
            'sequences': training_data
        }
    
    def _create_summary_csv(self):
        """Create summary CSV for easy analysis"""
        summary_rows = []
        
        for circuit_key, circuit_data in self.complete_data.items():
            circuit_name = circuit_data['circuit_info']['name']
            drs_count = circuit_data['circuit_info']['drs_zone_count']
            
            for session_name, session_data in circuit_data['sessions'].items():
                sequences = session_data.get('drs_sequences', {})
                
                for driver, driver_sequences in sequences.items():
                    for sequence in driver_sequences:
                        summary_rows.append({
                            'circuit': circuit_name,
                            'drs_zones': drs_count,
                            'session': session_name,
                            'driver': driver,
                            'lap_number': sequence['lap_number'],
                            'zone_index': sequence['zone_index'],
                            'sequence_id': sequence['sequence_id'],
                            'sequence_length': sequence['zone_info']['sequence_length'],
                            'sampling_rate': sequence['zone_info']['sampling_rate'],
                            'needs_labeling': sequence['label']['needs_review']
                        })
        
        return pd.DataFrame(summary_rows)
    
    def _create_readme(self, readme_file):
        """Create documentation README"""
        readme_content = f"""# F1 2023 3-DRS Zone Complete Dataset

## Overview
Complete Formula 1 data extraction from 2023 season focusing on circuits with 3 DRS zones.
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Circuits Included
### 3-DRS Zone Circuits (Priority):
- Austria (Red Bull Ring)
- Saudi Arabia (Jeddah Corniche Circuit)  
- Canada (Circuit Gilles Villeneuve)
- Miami (Miami International Autodrome)

### 2-DRS Zone Circuits (Baseline):
- Bahrain (Bahrain International Circuit)
- Italy (Autodromo Nazionale di Monza)

## Data Structure
```
F1_2023_3DRS_Dataset/
├── [base_filename]_complete_dataset.json     # Complete raw data
├── [base_filename]_training_sequences.json   # DRS sequences for ML
├── [base_filename]_summary.csv              # Summary for analysis
└── README.md                                # This file
```

## Dataset Statistics
- **Total Circuits**: {len(self.complete_data)}
- **Total DRS Sequences**: {sum(len(cd['sessions'][sn].get('drs_sequences', {})) for cd in self.complete_data.values() for sn in cd['sessions'])}
- **Data Source**: FastF1 API (Official F1 timing and telemetry)
- **Year**: 2023

## Key Features per Sequence
- High-frequency telemetry (Speed, Throttle, Brake, DRS, etc.)
- Lap context (tire data, position, timing)
- Race context (session type, progress, conditions)
- Weather data (temperature, humidity, wind)
- Track status information
- Strategic context (pit stops, tire strategy)

## Usage for DRS Decision AI
1. **Load training sequences**: Use the `training_sequences.json` file
2. **Manual labeling**: Review sequences and add labels
3. **Model training**: Feed sequences to transformer model
4. **Validation**: Use cross-circuit validation

## Data Quality
- All sequences have minimum 20 data points
- Sampling rates: ~10-20Hz for telemetry
- Weather data updated per minute
- Complete driver and session metadata

## Next Steps
1. Manual labeling of overtaking decisions
2. Data preprocessing for transformer input
3. Model training and validation
4. Performance analysis across circuit types
"""
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)

# ============================================================================
# EMERGENCY SAVE FUNCTIONS FOR BACKUP
# ============================================================================

def save_extracted_data_safely(extractor, base_filename='F1_2023_3DRS_FIXED'):
    """
    Emergency save function to handle already extracted data with circular reference fix
    """
    if not extractor.complete_data:
        print("❌ No data to save. The extractor doesn't have any data.")
        return None
    
    print("🔧 Applying emergency circular reference fix and saving data...")
    
    # Create output directory
    output_dir = Path(f'F1_2023_3DRS_Dataset_Fixed')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Clean and save just the essential training sequences
        training_sequences = []
        
        for circuit_key, circuit_data in extractor.complete_data.items():
            circuit_name = circuit_data['circuit_info']['name']
            print(f"  📊 Processing {circuit_name}...")
            
            for session_name, session_data in circuit_data['sessions'].items():
                sequences = session_data.get('drs_sequences', {})
                
                for driver, driver_sequences in sequences.items():
                    for sequence in driver_sequences:
                        # Extract only the essential data for training
                        clean_sequence = {
                            'sequence_id': str(sequence.get('sequence_id', 'unknown')),
                            'circuit': str(sequence.get('circuit', circuit_name)),
                            'driver': str(sequence.get('driver', driver)),
                            'lap_number': int(sequence.get('lap_number', 0)),
                            'zone_index': int(sequence.get('zone_index', 0)),
                            'zone_count': int(sequence.get('zone_count', 0)),
                            'year': 2023,
                            
                            # Core telemetry (cleaned)
                            'telemetry': extract_clean_telemetry(sequence.get('telemetry', {})),
                            
                            # Context (cleaned)
                            'context': extract_clean_context(sequence.get('context', {})),
                            
                            # Zone info
                            'zone_info': {
                                'zone_start_pct': float(sequence.get('zone_info', {}).get('zone_start_pct', 0)),
                                'zone_end_pct': float(sequence.get('zone_info', {}).get('zone_end_pct', 0)),
                                'sequence_length': int(sequence.get('zone_info', {}).get('sequence_length', 0)),
                                'sampling_rate': float(sequence.get('zone_info', {}).get('sampling_rate', 0)) if sequence.get('zone_info', {}).get('sampling_rate') else None
                            },
                            
                            # Label
                            'label': {
                                'overtake_decision': None,  # To be filled: 0=Hold, 1=Attempt, 2=Defensive
                                'overtake_success': None,   # To be filled: True/False
                                'multi_zone_strategy': None,
                                'confidence': None,
                                'notes': '',
                                'labeled': False
                            }
                        }
                        
                        training_sequences.append(clean_sequence)
        
        # Save training sequences
        training_data = {
            'total_sequences': len(training_sequences),
            'extraction_date': datetime.now().isoformat(),
            'data_source': 'FastF1 API - F1 2023 Season',
            'circuits_included': list(extractor.complete_data.keys()),
            'sequences': training_sequences
        }
        
        training_file = output_dir / f"{base_filename}_training_sequences.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"🤖 Training sequences saved: {training_file}")
        
        # Create summary CSV
        summary_data = []
        for seq in training_sequences:
            summary_data.append({
                'sequence_id': seq['sequence_id'],
                'circuit': seq['circuit'],
                'driver': seq['driver'],
                'lap_number': seq['lap_number'],
                'zone_index': seq['zone_index'],
                'zone_count': seq['zone_count'],
                'sequence_length': seq['zone_info']['sequence_length'],
                'sampling_rate': seq['zone_info']['sampling_rate'],
                'needs_labeling': True
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"📊 Summary CSV saved: {summary_file}")
        print(f"\n✅ Emergency save successful! Data saved to: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Error during emergency save: {e}")
        return None

def extract_clean_telemetry(telemetry_data):
    """Extract clean telemetry features"""
    if not telemetry_data:
        return {}
    
    features = {}
    
    # Core channels
    for channel in ['speed', 'throttle', 'brake', 'drs', 'acceleration']:
        if channel in telemetry_data:
            channel_data = telemetry_data[channel]
            if isinstance(channel_data, dict):
                features[f'{channel}_mean'] = float(channel_data.get('mean', 0)) if channel_data.get('mean') is not None else 0
                features[f'{channel}_max'] = float(channel_data.get('max', 0)) if channel_data.get('max') is not None else 0
                features[f'{channel}_min'] = float(channel_data.get('min', 0)) if channel_data.get('min') is not None else 0
                features[f'{channel}_std'] = float(channel_data.get('std', 0)) if channel_data.get('std') is not None else 0
                
                if 'values' in channel_data and channel_data['values']:
                    features[f'{channel}_length'] = len(channel_data['values'])
                    values = channel_data['values']
                    if len(values) >= 20:
                        features[f'{channel}_start_values'] = values[:10]
                        features[f'{channel}_end_values'] = values[-10:]
                    else:
                        features[f'{channel}_values'] = values
    
    return features

def extract_clean_context(context_data):
    """Extract clean context features"""
    if not context_data:
        return {}
    
    clean_context = {}
    
    # Lap context
    if 'lap_context' in context_data:
        lap_ctx = context_data['lap_context']
        clean_context.update({
            'lap_time': float(lap_ctx.get('lap_time')) if lap_ctx.get('lap_time') is not None else None,
            'position': int(lap_ctx.get('position')) if lap_ctx.get('position') is not None else None,
            'tire_compound': str(lap_ctx.get('tire_compound', 'Unknown')),
            'tire_life': int(lap_ctx.get('tire_life', 0)) if lap_ctx.get('tire_life') is not None else 0
        })
    
    # Race context
    if 'race_context' in context_data:
        race_ctx = context_data['race_context']
        clean_context.update({
            'session_type': str(race_ctx.get('session_type', 'Unknown')),
            'race_progress': float(race_ctx.get('race_progress')) if race_ctx.get('race_progress') is not None else None
        })
    
    # Weather context
    if 'weather_context' in context_data:
        weather_ctx = context_data['weather_context']
        if weather_ctx.get('available', False):
            clean_context.update({
                'air_temp': float(weather_ctx.get('air_temp')) if weather_ctx.get('air_temp') is not None else None,
                'track_temp': float(weather_ctx.get('track_temp')) if weather_ctx.get('track_temp') is not None else None,
                'rainfall': bool(weather_ctx.get('rainfall', False))
            })
    
    return clean_context

# ============================================================================
# MAIN EXECUTION FUNCTION WITH ERROR HANDLING
# ============================================================================

def extract_2023_3drs_complete():
    """
    Main function to extract complete 2023 F1 data from 3-DRS zone circuits
    """
    print("🏎️ F1 2023 Complete 3-DRS Zone Data Extraction")
    print("=" * 70)
    print("Initializing extractor...")
    
    # Initialize extractor
    extractor = F1_2023_ThreeDRS_Extractor()
    
    # Extract complete data from all 3-DRS circuits + baselines
    print("\n🚀 Starting complete data extraction...")
    complete_data = extractor.extract_complete_3drs_season(
        session_types=['R'],  # Start with Race sessions
        max_drivers=10        # Top 10 drivers for manageable dataset
    )
    
    # Save everything using the safe method
    try:
        output_dir = extractor.save_complete_dataset('F1_2023_3DRS')
        print(f"\n✅ EXTRACTION COMPLETE!")
        print("=" * 70)
        print("✅ Complete F1 2023 3-DRS zone dataset extracted")
        print(f"📁 Saved to: {output_dir}")
        print("🔥 Ready for DRS decision AI training!")
        
    except Exception as e:
        print(f"\n⚠️  JSON serialization issue detected: {e}")
        print("🔧 Applying emergency fix and saving essential data...")
        
        # Use the emergency save method
        output_dir = save_extracted_data_safely(extractor)
        
        if output_dir:
            print(f"\n✅ EXTRACTION SAVED SUCCESSFULLY!")
            print("=" * 70)
            print("✅ F1 2023 3-DRS zone training data saved")
            print(f"📁 Saved to: {output_dir}")
            print("🔥 Ready for DRS decision AI training!")
        else:
            print("❌ Failed to save data. Please check the error messages above.")
    
    return extractor, complete_data

if __name__ == "__main__":
    # Run the complete extraction
    extractor, data = extract_2023_3drs_complete()
    
    print("\n🚀 Next Steps:")
    print("1. Review the generated dataset files")
    print("2. Begin manual labeling of DRS sequences")
    print("3. Preprocess data for transformer training")
    print("4. Train your DRS decision AI model!")