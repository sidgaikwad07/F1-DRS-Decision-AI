"""
Created on Wed Jun 25 17:00:24 2025

@author: sid

2025 Austrian GP AI Predictions - Red Bull Ring Analysis
Advanced F1 AI system for predicting the Austrian Grand Prix

Current 2025 Season Context:
- Championship Leader: Oscar Piastri (McLaren)
- P2: Lando Norris (+22 points behind Piastri)  
- P3: Max Verstappen (+43 points behind Piastri)
- Recent: George Russell won Canadian GP (Mercedes)
- Date: June 29, 2025 | Track: Red Bull Ring, Spielberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AustriaGP2025Context:
    """Current 2025 F1 season context and Red Bull Ring characteristics"""
    
    def __init__(self):
        # 2025 Championship standings (current) - CORRECTED TEAMS
        self.championship_standings = {
            'drivers': {
                'piastri': {'points': 198, 'team': 'mclaren', 'position': 1},
                'norris': {'points': 176, 'team': 'mclaren', 'position': 2},
                'verstappen': {'points': 155, 'team': 'red_bull', 'position': 3},
                'russell': {'points': 136, 'team': 'mercedes', 'position': 4},
                'hamilton': {'points': 79, 'team': 'ferrari', 'position': 6},  # NOW AT FERRARI
                'leclerc': {'points': 104, 'team': 'ferrari', 'position': 5},
                'sainz': {'points': 13, 'team': 'williams', 'position': 13},    # NOW AT WILLIAMS
                'alonso': {'points': 8, 'team': 'aston_martin', 'position': 16},
                'antonelli': {'points': 63, 'team': 'mercedes', 'position': 7}, # REPLACES HAMILTON
                'tsunoda': {'points': 10, 'team': 'red_bull', 'position': 15},   # VERSTAPPEN'S TEAMMATE
                'albon' : {'points' : 42, 'team' : 'williams', 'position' : 42}
            },
            'constructors': {
                'mclaren': 374,      # Leading by large margin
                'ferrari': 183,      # Hamilton boost + good car
                'red_bull': 162,     # Home race pressure + Verstappen
                'mercedes': 199,     # Russell wins + Antonelli learning
                'williams': 55,      # Sainz boost + Albon solid
                'aston_martin': 22
            }
        }
        
        # Red Bull Ring characteristics
        self.track_data = {
            'name': 'Red Bull Ring',
            'location': 'Spielberg, Austria',
            'length_km': 4.318,
            'corners': 10,
            'drs_zones': 3,
            'lap_time_2024': '1:03.314',  # Verstappen pole
            'altitude_m': 678,
            'track_type': 'power_circuit',
            'overtaking_difficulty': 0.3,  # Relatively easy
            'red_bull_home_advantage': 0.2,  # 20% boost for RB
            'key_corners': ['Turn 1', 'Turn 3 (Remus)', 'Turn 4', 'Turn 9 (Rindt)'],
            'weather_forecast': {
                'temperature': 30,  # Celsius
                'conditions': 'sunny',
                'rain_probability': 0.1,
                'wind_speed': 15  # km/h
            }
        }
        
        # 2025 season momentum and form - UPDATED WITH ALL DRIVERS
        self.recent_form = {
            'piastri': {'canada': 'DNF_collision', 'momentum': 0.88, 'championship_leader': 0.95},
            'norris': {'canada': 'retired', 'momentum': 0.65, 'pressure': 0.8, 'teammate_clash': -0.1},
            'verstappen': {'canada': 'podium', 'momentum': 0.85, 'home_boost': 0.95, 'comeback_mode': 0.9},
            'russell': {'canada': 'win', 'momentum': 0.95, 'confidence': 0.9},
            'hamilton': {'canada': 'points', 'momentum': 0.80, 'ferrari_motivation': 0.9, 'new_team': 0.85},
            'leclerc': {'momentum': 0.78, 'hamilton_boost': 0.85, 'austria_history': 0.8},
            'sainz': {'momentum': 0.72, 'williams_adaptation': 0.7, 'consistency': 0.85},
            'alonso': {'momentum': 0.78, 'veteran_experience': 0.95, 'aston_martin': 0.7},
            'antonelli': {'momentum': 0.70, 'rookie_season': 0.6, 'learning': 0.8},
            'tsunoda': {'momentum': 0.75, 'red_bull_promotion': 0.8, 'verstappen_teammate': 0.7}
        }
        
        # Verstappen penalty points situation
        self.verstappen_penalty_status = {
            'current_points': 11,
            'ban_threshold': 12,
            'expiring_after_austria': 2,  # Will drop to 9 points
            'pressure_factor': 0.1  # Slight pressure but manageable
        }

class RedBullRingPredictor:
    """Advanced AI predictor for Red Bull Ring racing scenarios"""
    
    def __init__(self, context: AustriaGP2025Context):
        self.context = context
        self.track = context.track_data
        self.standings = context.championship_standings
        self.form = context.recent_form
        
        # Driver performance profiles at Red Bull Ring - COMPLETE 2025 GRID
        self.driver_track_affinity = {
            'verstappen': 0.96,  # 5 wins here, home race + track master
            'piastri': 0.91,     # Championship leader form + excellent driver
            'norris': 0.89,      # 2nd in championship but can be inconsistent
            'hamilton': 0.88,    # Ferrari motivation + power circuit master
            'russell': 0.86,     # Recent winner + good pace
            'leclerc': 0.83,     # Good Ferrari driver
            'alonso': 0.85,      # Veteran experience
            'sainz': 0.77,       # Adapting to Williams but experienced
            'antonelli': 0.73,   # Rookie learning but talented
            'tsunoda': 0.74      # Decent but new to Red Bull pressure
        }
        
        # Team car performance at Red Bull Ring type - UPDATED FOR 2025 REALITY
        self.team_performance = {
            'mclaren': 0.95,     # Dominant 2025 car + championship leading
            'red_bull': 0.90,    # Still strong + home track advantage
            'ferrari': 0.87,     # Hamilton effect + improved car
            'mercedes': 0.80,    # Rebuilding phase but Russell wins
            'williams': 0.74,    # Improving with Sainz
            'aston_martin': 0.71 # Mid-tier performance
        }
    
    def predict_qualifying_results(self) -> Dict:
        """Predict qualifying results with realistic 2025 season context"""
        print("🏁 PREDICTING 2025 AUSTRIAN GP QUALIFYING")
        print("=" * 50)
        
        # Calculate qualifying pace for each driver - MORE REALISTIC
        driver_qualifying_scores = {}
        
        for driver, driver_data in self.standings['drivers'].items():
            if driver not in self.driver_track_affinity:
                continue
                
            team = driver_data['team']
            
            # More realistic base factors
            base_score = 85.0  # Lower base to allow more differentiation
            
            # Track affinity (more important)
            track_bonus = self.driver_track_affinity[driver] * 20
            
            # Team car performance (much more important)
            car_performance = self.team_performance.get(team, 0.75) * 25
            
            # Current championship form (very important)
            championship_position = driver_data['position']
            championship_bonus = max(0, (10 - championship_position) * 2)  # Top drivers get bonus
            
            # Driver-specific realistic bonuses
            if driver == 'verstappen':  # Home race + track master
                pressure_effect = +8
            elif driver == 'piastri':  # Championship leader + best car
                pressure_effect = +7
            elif driver == 'norris':  # 2nd in championship + best car
                pressure_effect = +6
            elif driver == 'hamilton':  # Ferrari motivation but adapting
                pressure_effect = +4
            elif driver == 'russell':  # Good form but not title contender
                pressure_effect = +3
            elif driver == 'leclerc':  # Ferrari + good driver
                pressure_effect = +3
            elif driver == 'tsunoda':  # New to Red Bull, less experienced
                pressure_effect = -2
            elif driver == 'antonelli':  # Rookie
                pressure_effect = -5
            else:
                pressure_effect = 0
            
            # Realistic qualifying variability (smaller range)
            qualifying_luck = np.random.normal(0, 1.5)  # Reduced randomness
            
            total_score = (base_score + track_bonus + car_performance + 
                          championship_bonus + pressure_effect + qualifying_luck)
            
            driver_qualifying_scores[driver] = total_score
        
        # Sort by score (highest = pole position)
        sorted_drivers = sorted(driver_qualifying_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Create qualifying results
        qualifying_results = []
        for position, (driver, score) in enumerate(sorted_drivers, 1):
            team = self.standings['drivers'][driver]['team']
            qualifying_results.append({
                'position': position,
                'driver': driver.capitalize(),
                'team': team.replace('_', ' ').title(),
                'predicted_time': self._calculate_lap_time(score, position),
                'gap_to_pole': self._calculate_gap_to_pole(score, sorted_drivers[0][1])
            })
        
        return {
            'results': qualifying_results,
            'pole_position': sorted_drivers[0][0].capitalize(),
            'front_row': [sorted_drivers[0][0].capitalize(), sorted_drivers[1][0].capitalize()],
            'top_3': [driver.capitalize() for driver, _ in sorted_drivers[:3]],
            'surprises': self._identify_qualifying_surprises(qualifying_results)
        }
    
    def predict_race_outcome(self, qualifying_results: Dict) -> Dict:
        """Predict realistic race outcome based on qualifying and race factors"""
        print("\n🏎️ PREDICTING 2025 AUSTRIAN GP RACE OUTCOME")
        print("=" * 50)
        
        # Initialize race simulation with more realistic factors
        race_factors = {}
        
        for result in qualifying_results['results']:
            driver = result['driver'].lower()
            if driver not in self.driver_track_affinity:
                continue
                
            starting_pos = result['position']
            team = result['team'].lower().replace(' ', '_')
            
            # More realistic race day factors
            base_race_score = 90.0  # Realistic base
            
            # Starting position advantage (less dramatic)
            grid_advantage = max(0, (11 - starting_pos)) * 1.5
            
            # Race craft and overtaking ability (more important)
            racecraft = self.driver_track_affinity[driver] * 15
            
            # Team strategy and car performance (very important)
            car_race_pace = self.team_performance.get(team, 0.75) * 20
            
            # Championship standings influence (important)
            championship_position = self.standings['drivers'][driver]['position']
            championship_factor = max(0, (11 - championship_position) * 1.5)
            
            # Driver-specific race factors
            if driver == 'verstappen':
                home_race_boost = 6  # Strong but not overwhelming
                title_intensity = 5  # Comeback motivation
            elif driver == 'piastri':
                championship_confidence = 5  # Leader confidence
                title_intensity = 4  # Slight pressure
            elif driver == 'norris':
                recovery_motivation = 4  # Bounce back from Canada
                title_intensity = 5  # Close championship gap
            elif driver == 'hamilton':
                ferrari_motivation = 4  # New team energy
                title_intensity = 2
            elif driver == 'russell':
                momentum_boost = 3  # Canada win confidence
                title_intensity = 2
            else:
                title_intensity = 1
                
            # Weather handling and tire management
            tire_management = {
                'hamilton': 5, 'alonso': 5, 'verstappen': 4.5,
                'piastri': 4, 'russell': 4, 'leclerc': 3.5,
                'norris': 3.5, 'sainz': 4, 'tsunoda': 3, 'antonelli': 2.5
            }.get(driver, 3.0)
            
            # Realistic incident risk (much lower)
            incident_risk = {
                'verstappen': 0.02, 'hamilton': 0.02, 'russell': 0.03,
                'piastri': 0.03, 'leclerc': 0.04, 'sainz': 0.04,
                'norris': 0.06, 'alonso': 0.02, 'tsunoda': 0.07, 'antonelli': 0.08
            }.get(driver, 0.05)
            
            # Apply incident risk (much smaller penalty)
            if np.random.random() < incident_risk:
                incident_penalty = -8  # Smaller penalty
            else:
                incident_penalty = 0
            
            # Small random factor
            race_luck = np.random.normal(0, 1.5)
            
            # Calculate total race score
            total_race_score = (base_race_score + grid_advantage + racecraft + 
                              car_race_pace + championship_factor + 
                              title_intensity + tire_management + incident_penalty + race_luck)
            
            # Add specific driver bonuses
            if driver == 'verstappen':
                total_race_score += 6  # Home advantage
            elif driver in ['piastri', 'norris']:
                total_race_score += 4  # Best car advantage
            elif driver == 'hamilton':
                total_race_score += 4  # Ferrari motivation
            
            race_factors[driver] = {
                'total_score': total_race_score,
                'starting_position': starting_pos,
                'incident_occurred': incident_penalty < 0
            }
        
        # Sort by race score
        race_classification = sorted(race_factors.items(), 
                                   key=lambda x: x[1]['total_score'], reverse=True)
        
        # Create race results
        race_results = []
        for position, (driver, data) in enumerate(race_classification, 1):
            team = self.standings['drivers'][driver]['team']
            points = self._calculate_f1_points(position)
            
            race_results.append({
                'position': position,
                'driver': driver.capitalize(),
                'team': team.replace('_', ' ').title(),
                'starting_position': data['starting_position'],
                'positions_gained': data['starting_position'] - position,
                'points': points,
                'incident': data['incident_occurred']
            })
        
        return {
            'results': race_results,
            'winner': race_classification[0][0].capitalize(),
            'podium': [driver.capitalize() for driver, _ in race_classification[:3]],
            'points_finishers': [r for r in race_results if r['points'] > 0],
            'biggest_movers': self._identify_biggest_movers(race_results),
            'championship_impact': self._calculate_championship_impact(race_results)
        }
    
    def predict_drs_strategy_scenarios(self, race_predictions: Dict) -> Dict:
        """Predict detailed DRS usage strategies for each driver at Red Bull Ring"""
        print("\n🚀 PREDICTING DRS STRATEGIC DECISIONS")
        print("=" * 45)
        
        # Red Bull Ring DRS zones analysis
        drs_zones = {
            'main_straight': {
                'length_m': 800,
                'speed_gain': '15-20 km/h',
                'overtaking_probability': 0.75,
                'detection_point': 'Turn 10 exit',
                'activation_point': 'Start/finish straight',
                'key_for': ['Verstappen', 'Piastri', 'Norris', 'Hamilton']
            },
            'zone_2': {
                'length_m': 650,
                'speed_gain': '12-15 km/h', 
                'overtaking_probability': 0.60,
                'detection_point': 'Turn 1 exit',
                'activation_point': 'Before Turn 3',
                'key_for': ['Russell', 'Leclerc', 'Sainz']
            },
            'zone_3': {
                'length_m': 500,
                'speed_gain': '8-12 km/h',
                'overtaking_probability': 0.45,
                'detection_point': 'Turn 6 exit', 
                'activation_point': 'Turns 7-8 complex',
                'key_for': ['Alonso', 'Tsunoda', 'Antonelli']
            }
        }
        
        # Driver-specific DRS strategies
        driver_drs_strategies = {}
        
        for result in race_predictions['results'][:10]:
            driver = result['driver'].lower()
            position = result['position']
            team = result['team'].lower().replace(' ', '_')
            
            # Base DRS effectiveness for each driver
            drs_effectiveness = {
                'verstappen': 0.95,  # Master of DRS timing
                'hamilton': 0.93,   # Exceptional DRS usage
                'piastri': 0.89,    # Learning but effective
                'russell': 0.87,    # Good DRS management
                'norris': 0.85,     # Sometimes impatient
                'leclerc': 0.84,    # Can be aggressive
                'alonso': 0.92,     # Veteran DRS wisdom
                'sainz': 0.82,      # Adapting to Williams
                'antonelli': 0.75,  # Rookie learning DRS
                'tsunoda': 0.80     # Decent but inconsistent
            }.get(driver, 0.80)
            
            # DRS strategic scenarios for this driver
            scenarios = self._generate_drs_scenarios(driver, position, team, drs_effectiveness)
            
            driver_drs_strategies[driver] = {
                'effectiveness_rating': drs_effectiveness,
                'primary_zone': self._get_primary_drs_zone(driver, position),
                'offensive_scenarios': scenarios['offensive'],
                'defensive_scenarios': scenarios['defensive'],
                'strategic_decisions': scenarios['strategic'],
                'risk_assessment': scenarios['risk']
            }
        
        return {
            'drs_zones': drs_zones,
            'driver_strategies': driver_drs_strategies,
            'key_battles': self._predict_drs_battles(driver_drs_strategies),
            'strategic_timeline': self._create_drs_timeline(),
            'effectiveness_ranking': self._rank_drs_effectiveness(driver_drs_strategies)
        }
    
    def _generate_drs_scenarios(self, driver: str, position: int, team: str, effectiveness: float) -> Dict:
        """Generate specific DRS scenarios for each driver"""
        
        # Offensive DRS scenarios (when to attack)
        offensive_scenarios = []
        
        if driver == 'verstappen':
            offensive_scenarios.extend([
                "Turn 3 DRS zone: Attack slow-starting cars on medium tires",
                "Main straight: Use home crowd energy for psychological advantage",
                "Late-race: Deploy DRS with fresh tires for championship points",
                "Safety car restart: Maximize DRS effectiveness in clean air"
            ])
        elif driver == 'piastri':
            offensive_scenarios.extend([
                "Championship mode: Use DRS conservatively to maintain points lead",
                "Main straight: Counter Verstappen attacks with McLaren top speed",
                "Mid-race: Strategic DRS to gap slower cars and control pace",
                "Tire undercut: Combine DRS with fresh rubber for position gain"
            ])
        elif driver == 'norris':
            offensive_scenarios.extend([
                "Recovery mode: Aggressive DRS usage to regain championship points",
                "McLaren teamwork: Use DRS to help Piastri or attack independently", 
                "Turn 1 zone: Exploit slipstream from Piastri if following",
                "Frustration factor: May overuse DRS and compromise exit speed"
            ])
        elif driver == 'hamilton':
            offensive_scenarios.extend([
                "Ferrari debut: Prove new car's DRS effectiveness vs old rivals",
                "Experience factor: Perfect DRS timing in wheel-to-wheel combat",
                "Power unit advantage: Use Ferrari straight-line speed with DRS",
                "Veteran moves: Late-braking DRS attacks when others expect defense"
            ])
        else:
            offensive_scenarios.extend([
                f"Position {position}: Use DRS to maintain/improve current standing",
                "Opportunistic: Attack when leaders battle each other",
                "Clean air: Maximize DRS in free practice and qualifying"
            ])
        
        # Defensive DRS scenarios (when NOT to use)
        defensive_scenarios = []
        
        if position <= 3:  # Leading group
            defensive_scenarios.extend([
                "Leading into Turn 1: Close DRS to prevent slipstream followers",
                "Final sector: Sacrifice DRS for better cornering speed",
                "Championship protection: Avoid risky DRS moves that invite contact",
                "Traffic management: Keep DRS closed when lapping backmarkers"
            ])
        elif position <= 6:  # Points battle
            defensive_scenarios.extend([
                "Points protection: Don't risk DRS moves that could end in gravel",
                "Tire preservation: Minimize DRS to reduce tire degradation",
                "Strategic patience: Wait for DRS detection opportunities"
            ])
        else:  # Back of grid
            defensive_scenarios.extend([
                "Blue flag situations: Use DRS to get out of leaders' way quickly",
                "Damage limitation: Conservative DRS to avoid further incidents"
            ])
        
        # Strategic decision matrix
        strategic_decisions = {
            'when_to_use_drs': [
                "Gap to car ahead: < 1 second",
                "Straight line speed advantage: Clear",
                "Tire condition: Equal or better than target",
                "Track position: Worth the risk",
                "Championship implications: Positive or neutral"
            ],
            'when_not_to_use_drs': [
                "Wet conditions: Reduced effectiveness + safety risk",
                "Tire degradation: Critical phase of stint",
                "Defensive position: Protecting from faster car behind", 
                "Technical issues: DRS system unreliable",
                "Championship protection: Leading with comfortable gap"
            ],
            'optimal_timing': [
                "Detection point: Latest possible for maximum effect",
                "Activation timing: Coordinate with slipstream",
                "Corner exit: Ensure clean activation",
                "Lap traffic: Avoid DRS usage in dirty air"
            ]
        }
        
        # Risk assessment
        risk_level = 'low'
        if driver in ['norris', 'leclerc']:  # Sometimes overaggressive
            risk_level = 'medium-high'
        elif driver in ['antonelli', 'tsunoda']:  # Learning/adapting
            risk_level = 'medium'
        elif driver in ['verstappen', 'hamilton', 'alonso']:  # Masters
            risk_level = 'low'
        
        return {
            'offensive': offensive_scenarios,
            'defensive': defensive_scenarios,
            'strategic': strategic_decisions,
            'risk': risk_level
        }
    
    def _get_primary_drs_zone(self, driver: str, position: int) -> str:
        """Determine which DRS zone is most important for each driver"""
        
        if driver in ['verstappen', 'piastri', 'norris']:  # Championship contenders
            return 'main_straight'  # Best overtaking opportunity
        elif position <= 6:  # Points contenders
            return 'zone_2'  # Good balance of speed and opportunity
        else:
            return 'zone_3'  # Take what you can get
    
    def _predict_drs_battles(self, strategies: Dict) -> List[Dict]:
        """Predict key DRS battles during the race"""
        
        key_battles = [
            {
                'battle': 'Piastri vs Verstappen',
                'zone': 'main_straight',
                'scenario': 'Championship leader vs home hero - DRS effectiveness crucial',
                'prediction': 'Verstappen slightly favored due to track knowledge',
                'laps': '15-25, 45-55',
                'importance': 'Championship defining'
            },
            {
                'battle': 'Hamilton vs Leclerc',
                'zone': 'main_straight + zone_2',
                'scenario': 'Ferrari teammates in different eras - DRS showcase',
                'prediction': 'Hamilton experience vs Leclerc aggression',
                'laps': '20-40',
                'importance': 'Team dynamics'
            },
            {
                'battle': 'Norris recovery drive',
                'zone': 'all_zones',
                'scenario': 'Championship challenger fighting back through field',
                'prediction': 'Aggressive DRS usage, high reward/risk',
                'laps': '10-60',
                'importance': 'Title fight impact'
            },
            {
                'battle': 'Russell vs midfield',
                'zone': 'zone_2',
                'scenario': 'Mercedes vs improving Williams/Aston Martin',
                'prediction': 'DRS efficiency vs raw pace',
                'laps': '25-50',
                'importance': 'Constructor points'
            }
        ]
        
        return key_battles
    
    def _create_drs_timeline(self) -> Dict:
        """Create lap-by-lap DRS strategic timeline"""
        
        timeline = {
            'laps_1_10': {
                'phase': 'Opening stint',
                'drs_strategy': 'Conservative - establish position',
                'key_drivers': ['Verstappen', 'Piastri'],
                'risk_level': 'Low',
                'notes': 'Full fuel loads, tire warming phase'
            },
            'laps_11_25': {
                'phase': 'First stint battle',
                'drs_strategy': 'Aggressive - make moves before pit stops',
                'key_drivers': ['Norris', 'Hamilton', 'Russell'],
                'risk_level': 'Medium',
                'notes': 'DRS most effective with lighter fuel'
            },
            'laps_26_40': {
                'phase': 'Pit window chaos',
                'drs_strategy': 'Strategic - undercut/overcut combinations',
                'key_drivers': ['All drivers'],
                'risk_level': 'High',
                'notes': 'Fresh tires vs track position decisions'
            },
            'laps_41_55': {
                'phase': 'Final stint setup',
                'drs_strategy': 'Calculated - championship implications',
                'key_drivers': ['Piastri', 'Verstappen', 'Norris'],
                'risk_level': 'Medium-High',
                'notes': 'Tire degradation affects DRS effectiveness'
            },
            'laps_56_71': {
                'phase': 'Championship finale',
                'drs_strategy': 'Maximum attack or protection mode',
                'key_drivers': ['Title contenders'],
                'risk_level': 'Maximum',
                'notes': 'Every point matters - DRS usage critical'
            }
        }
        
        return timeline
    
    def _rank_drs_effectiveness(self, strategies: Dict) -> List[Dict]:
        """Rank drivers by predicted DRS effectiveness"""
        
        effectiveness_ranking = []
        
        for driver, strategy in strategies.items():
            effectiveness_ranking.append({
                'driver': driver.capitalize(),
                'effectiveness': strategy['effectiveness_rating'],
                'primary_zone': strategy['primary_zone'],
                'risk_level': strategy['risk_assessment']
            })
        
        # Sort by effectiveness
        effectiveness_ranking.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        # FIXED: Added missing return statement
        return effectiveness_ranking
    
    def predict_strategic_scenarios(self) -> Dict:
        """Predict key strategic scenarios during the race"""
        print("\n⚡ PREDICTING STRATEGIC SCENARIOS")
        print("=" * 35)
        
        scenarios = {
            'pit_stop_strategies': {
                'one_stop': {
                    'probability': 0.7,
                    'best_for': ['Hamilton', 'Alonso', 'Sainz'],
                    'description': 'Medium-Hard compound strategy, lap 25-35 pit window'
                },
                'two_stop': {
                    'probability': 0.3,
                    'best_for': ['Verstappen', 'Norris', 'Leclerc'],
                    'description': 'Aggressive Medium-Hard-Medium strategy'
                }
            },
            'weather_impact': {
                'thunderstorm_risk': {
                    'probability': 0.1,
                    'impact': 'Could shuffle entire grid, favor experienced drivers'
                },
                'track_evolution': {
                    'probability': 1.0,
                    'description': 'Track temperature 30°C+ will increase tire degradation'
                }
            },
            'safety_car_probability': {
                'first_lap_incident': 0.15,
                'mid_race_mechanical': 0.25,
                'late_race_pressure': 0.10,
                'total_probability': 0.45
            }
        }
        
        return scenarios
    
    def predict_verstappen_specific(self) -> Dict:
        """Specific predictions for Verstappen's home race performance"""
        print("\n🇳🇱 VERSTAPPEN HOME RACE ANALYSIS")
        print("=" * 35)
        
        verstappen_prediction = {
            'grid_position_prediction': {
                'most_likely': 'P2-P3',
                'best_case': 'Pole Position',
                'worst_case': 'P5',
                'probability_top_3': 0.85
            },
            'race_outcome': {
                'podium_probability': 0.75,
                'win_probability': 0.35,
                'points_probability': 0.95,
                'dnf_probability': 0.05
            },
            'home_race_factors': {
                'crowd_support': 'Massive orange army expected',
                'pressure_handling': 0.95,  # Excellent under pressure
                'track_knowledge': 0.98,   # 5 previous wins
                'team_motivation': 0.9     # Red Bull home race
            },
            'championship_implications': {
                'points_gap_reduction': 'Likely to close gap to 25-30 points',
                'momentum_shift': 'Critical for championship fight',
                'pressure_on_mclaren': 'High - must respond to RB home win'
            },
            'penalty_points_impact': {
                'current_risk': 'Low (expires 2 points after race)',
                'racing_style': 'Slightly more conservative in wheel-to-wheel',
                'strategic_impact': 'Minimal - will still race aggressively'
            }
        }
        
        return verstappen_prediction
    
    def _calculate_lap_time(self, score: float, position: int) -> str:
        """Calculate predicted lap time based on score"""
        # Base time: 1:03.314 (2024 pole)
        base_time_seconds = 63.314
        
        # Score difference to time delta
        time_delta = (110 - score) * 0.05  # Each point = 0.05s
        
        predicted_time = base_time_seconds + time_delta
        
        # Convert to time format
        minutes = int(predicted_time // 60)
        seconds = predicted_time % 60
        
        return f"{minutes}:{seconds:06.3f}"
    
    def _calculate_gap_to_pole(self, score: float, pole_score: float) -> str:
        """Calculate gap to pole position"""
        if score == pole_score:
            return "POLE"
        
        gap_seconds = (pole_score - score) * 0.05
        
        if gap_seconds < 1.0:
            return f"+{gap_seconds:.3f}s"
        else:
            return f"+{gap_seconds:.2f}s"
    
    def _calculate_f1_points(self, position: int) -> int:
        """Calculate F1 championship points"""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        return points_system.get(position, 0)
    
    def _identify_qualifying_surprises(self, results: List[Dict]) -> List[str]:
        """Identify surprising qualifying performances based on 2025 expectations"""
        surprises = []
        
        for result in results:
            driver = result['driver'].lower()
            position = result['position']
            
            # Realistic expected positions based on 2025 championship standings and car performance
            expected_positions = {
                'piastri': 2,     # Championship leader, best car
                'norris': 3,      # 2nd in championship, best car  
                'verstappen': 1,  # Home track + excellent driver
                'russell': 5,     # Good driver, decent car
                'hamilton': 4,    # Great driver, good Ferrari
                'leclerc': 6,     # Good driver, Ferrari
                'sainz': 8,       # Good driver, Williams improving
                'alonso': 7,      # Excellent driver, mid-tier car
                'antonelli': 9,   # Rookie
                'tsunoda': 10     # Decent driver, new to Red Bull
            }
            
            expected = expected_positions.get(driver, 10)
            
            if position < expected - 2:
                surprises.append(f"{result['driver']} qualifies P{position} (better than expected P{expected})")
            elif position > expected + 2:
                surprises.append(f"{result['driver']} qualifies P{position} (worse than expected P{expected})")
        
        return surprises
    
    def _identify_biggest_movers(self, results: List[Dict]) -> List[str]:
        """Identify biggest position changes in race"""
        movers = []
        
        for result in results:
            gained = result['positions_gained']
            if gained >= 5:
                movers.append(f"{result['driver']}: +{gained} positions (P{result['starting_position']}→P{result['position']})")
            elif gained <= -5:
                movers.append(f"{result['driver']}: {gained} positions (P{result['starting_position']}→P{result['position']})")
        
        return movers
    
    def _calculate_championship_impact(self, results: List[Dict]) -> Dict:
        """Calculate impact on championship standings"""
        current_points = self.standings['drivers']
        new_standings = {}
        
        # Add race points to current standings
        for result in results:
            driver = result['driver'].lower()
            if driver in current_points:
                new_points = current_points[driver]['points'] + result['points']
                new_standings[driver] = new_points
        
        # Sort by points
        sorted_standings = sorted(new_standings.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate changes
        changes = {}
        for pos, (driver, points) in enumerate(sorted_standings, 1):
            old_position = current_points[driver]['position']
            position_change = old_position - pos
            
            changes[driver] = {
                'new_position': pos,
                'position_change': position_change,
                'new_points': points,
                'points_gained': points - current_points[driver]['points']
            }
        
        return {
            'new_standings': sorted_standings,
            'position_changes': changes,
            'title_fight': {
                'leader_gap': sorted_standings[1][1] - sorted_standings[0][1] if len(sorted_standings) > 1 else 0,
                'top_3_gap': sorted_standings[2][1] - sorted_standings[0][1] if len(sorted_standings) > 2 else 0
            }
        }

def create_prediction_visualization(qualifying_pred: Dict, race_pred: Dict, drs_pred: Dict, strategic_pred: Dict):
    """Create comprehensive visualization of predictions including DRS analysis"""
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # 1. Qualifying Results
    axes[0, 0].set_title('Predicted Qualifying Results', fontsize=14, fontweight='bold')
    
    drivers = [r['driver'] for r in qualifying_pred['results'][:8]]
    positions = [r['position'] for r in qualifying_pred['results'][:8]]
    colors = ['gold' if p == 1 else 'silver' if p == 2 else 'orange' if p == 3 else 'lightblue' for p in positions]
    
    bars = axes[0, 0].barh(range(len(drivers)), [9-i for i in range(len(drivers))], color=colors, alpha=0.7)
    axes[0, 0].set_yticks(range(len(drivers)))
    axes[0, 0].set_yticklabels([f"P{positions[i]} {drivers[i]}" for i in range(len(drivers))])
    axes[0, 0].set_xlabel('Predicted Performance Score')
    axes[0, 0].invert_yaxis()
    
    # 2. Race Results
    axes[0, 1].set_title('Predicted Race Results', fontsize=14, fontweight='bold')
    
    race_drivers = [r['driver'] for r in race_pred['results'][:8]]
    race_positions = [r['position'] for r in race_pred['results'][:8]]
    points = [r['points'] for r in race_pred['results'][:8]]
    
    bars = axes[0, 1].bar(range(len(race_drivers)), points, 
                         color=['gold' if p == 1 else 'silver' if p == 2 else 'orange' if p == 3 else 'lightgreen' for p in race_positions],
                         alpha=0.7)
    axes[0, 1].set_xticks(range(len(race_drivers)))
    axes[0, 1].set_xticklabels([f"{race_drivers[i]}\nP{race_positions[i]}" for i in range(len(race_drivers))], rotation=45)
    axes[0, 1].set_ylabel('Championship Points')
    
    # Add point values on bars
    for i, (bar, point) in enumerate(zip(bars, points)):
        if point > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            str(point), ha='center', va='bottom', fontweight='bold')
    
    # 3. DRS Effectiveness Ranking
    axes[0, 2].set_title('DRS Effectiveness Ranking', fontsize=14, fontweight='bold')
    
    drs_ranking = drs_pred['effectiveness_ranking'][:8]
    drs_drivers = [r['driver'] for r in drs_ranking]
    drs_scores = [r['effectiveness'] for r in drs_ranking]
    
    bars = axes[0, 2].barh(range(len(drs_drivers)), drs_scores, color='cyan', alpha=0.7)
    axes[0, 2].set_yticks(range(len(drs_drivers)))
    axes[0, 2].set_yticklabels(drs_drivers)
    axes[0, 2].set_xlabel('DRS Effectiveness Score')
    axes[0, 2].invert_yaxis()
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, drs_scores)):
        axes[0, 2].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{score:.2f}', ha='left', va='center', fontweight='bold')
    
    # 4. Position Changes
    axes[0, 3].set_title('Predicted Position Changes', fontsize=14, fontweight='bold')
    
    changes = [r['positions_gained'] for r in race_pred['results'][:8]]
    change_colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' for c in changes]
    
    bars = axes[0, 3].barh(range(len(race_drivers)), changes, color=change_colors, alpha=0.7)
    axes[0, 3].set_yticks(range(len(race_drivers)))
    axes[0, 3].set_yticklabels(race_drivers)
    axes[0, 3].set_xlabel('Positions Gained/Lost')
    axes[0, 3].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 3].invert_yaxis()
    
    # 5. DRS Zone Analysis
    axes[1, 0].set_title('DRS Zones at Red Bull Ring', fontsize=14, fontweight='bold')
    
    zones = list(drs_pred['drs_zones'].keys())
    zone_lengths = [drs_pred['drs_zones'][zone]['length_m'] for zone in zones]
    overtake_probs = [drs_pred['drs_zones'][zone]['overtaking_probability'] for zone in zones]
    
    bars = axes[1, 0].bar(zones, zone_lengths, color='purple', alpha=0.7)
    axes[1, 0].set_ylabel('Zone Length (meters)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add overtaking probability as text
    for i, (bar, prob) in enumerate(zip(bars, overtake_probs)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                        f'{prob:.0%} overtake', ha='center', va='bottom', fontweight='bold')
    
    # 6. Strategic Scenarios
    axes[1, 1].set_title('Strategic Scenario Probabilities', fontsize=14, fontweight='bold')
    
    scenarios = ['One-Stop Strategy', 'Two-Stop Strategy', 'Safety Car', 'Thunderstorm']
    probabilities = [0.7, 0.3, 0.45, 0.1]
    
    bars = axes[1, 1].bar(scenarios, probabilities, color='lightcoral', alpha=0.7)
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    for bar, prob in zip(bars, probabilities):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{prob:.0%}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Championship Impact
    axes[1, 2].set_title('Championship Standings After Race', fontsize=14, fontweight='bold')
    
    champ_impact = race_pred['championship_impact']
    top_drivers = list(champ_impact['new_standings'][:5])
    new_points = [points for driver, points in top_drivers]
    driver_names = [driver.capitalize() for driver, points in top_drivers]
    
    bars = axes[1, 2].bar(range(len(driver_names)), new_points, 
                         color=['gold', 'silver', 'orange', 'lightblue', 'lightgreen'], alpha=0.7)
    axes[1, 2].set_xticks(range(len(driver_names)))
    axes[1, 2].set_xticklabels(driver_names, rotation=45)
    axes[1, 2].set_ylabel('Total Championship Points')
    
    # 8. DRS Battle Timeline
    axes[1, 3].set_title('DRS Battle Timeline', fontsize=14, fontweight='bold')
    
    battles = drs_pred['key_battles'][:4]
    battle_names = [b['battle'].split(' vs ')[0] for b in battles]
    importance_scores = {'Championship defining': 10, 'Team dynamics': 7, 'Title fight impact': 9, 'Constructor points': 5}
    importance_values = [importance_scores.get(b['importance'], 5) for b in battles]
    
    bars = axes[1, 3].bar(range(len(battle_names)), importance_values, color='red', alpha=0.7)
    axes[1, 3].set_xticks(range(len(battle_names)))
    axes[1, 3].set_xticklabels(battle_names, rotation=45)
    axes[1, 3].set_ylabel('Battle Importance Score')
    
    plt.suptitle('2025 AUSTRIAN GP AI PREDICTIONS - WITH DRS STRATEGY ANALYSIS', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def run_austria_gp_2025_predictions():
    """
    Run comprehensive 2025 Austrian GP predictions
    """
    print("🏁 2025 AUSTRIAN GP AI PREDICTION SYSTEM")
    print("🇦🇹 Red Bull Ring, Spielberg | June 29, 2025")
    print("=" * 60)
    print("🔄 UPDATED WITH CORRECT 2025 DRIVER LINEUP:")
    print("   🔴 Hamilton → Ferrari | 🔵 Sainz → Williams")
    print("   🔵 Antonelli → Mercedes | 🔴 Tsunoda → Red Bull")
    
    # Initialize prediction system
    context = AustriaGP2025Context()
    predictor = RedBullRingPredictor(context)
    
    # Current championship context
    print("\n📊 CURRENT 2025 CHAMPIONSHIP STANDINGS:")
    print(f"   1. Oscar Piastri (McLaren): 155 points")
    print(f"   2. Lando Norris (McLaren): 133 points (+22)")
    print(f"   3. Max Verstappen (Red Bull): 112 points (+43)")
    print(f"   4. George Russell (Mercedes): 98 points")
    print(f"   5. Lewis Hamilton (Ferrari): 87 points")  # CORRECTED TEAM
    
    print(f"\n🏆 CONSTRUCTORS CHAMPIONSHIP:")
    print(f"   1. McLaren: 288 points (DOMINANT)")
    print(f"   2. Ferrari: 166 points (Hamilton boost)")
    print(f"   3. Red Bull: 134 points (HOME RACE!)")
    print(f"   4. Mercedes: 126 points (Russell momentum)")
    print(f"   5. Williams: 83 points (Sainz effect)")
    
    # Run predictions
    qualifying_predictions = predictor.predict_qualifying_results()
    race_predictions = predictor.predict_race_outcome(qualifying_predictions)
    drs_predictions = predictor.predict_drs_strategy_scenarios(race_predictions)
    strategic_predictions = predictor.predict_strategic_scenarios()
    verstappen_analysis = predictor.predict_verstappen_specific()
    
    # Display results
    print(f"\n🏁 QUALIFYING PREDICTIONS:")
    print("-" * 30)
    for result in qualifying_predictions['results'][:8]:  # Show top 8
        print(f"   P{result['position']}: {result['driver']} ({result['team']}) - {result['predicted_time']}")
    
    print(f"\n🏆 RACE PREDICTIONS:")
    print("-" * 20)
    for result in race_predictions['results'][:8]:  # Show top 8
        change = result['positions_gained']
        change_str = f"(+{change})" if change > 0 else f"({change})" if change < 0 else "(0)"
        print(f"   P{result['position']}: {result['driver']} - {result['points']} pts {change_str}")
    
    print(f"\n🥇 PODIUM PREDICTION: {', '.join(race_predictions['podium'])}")
    print(f"🏆 RACE WINNER: {race_predictions['winner']}")
    
    print(f"\n🇳🇱 VERSTAPPEN HOME RACE ANALYSIS:")
    print("-" * 35)
    print(f"   Grid Position: {verstappen_analysis['grid_position_prediction']['most_likely']}")
    print(f"   Win Probability: {verstappen_analysis['race_outcome']['win_probability']:.0%}")
    print(f"   Podium Probability: {verstappen_analysis['race_outcome']['podium_probability']:.0%}")
    print(f"   Home Advantage: 95% track affinity + crowd support")
    
    print(f"\n🚀 DRS STRATEGIC ANALYSIS:")
    print("-" * 30)
    
    # DRS Zone Information
    print(f"🏁 RED BULL RING DRS ZONES:")
    for zone_name, zone_data in drs_predictions['drs_zones'].items():
        print(f"   {zone_name.upper()}: {zone_data['length_m']}m, {zone_data['speed_gain']} gain")
    
    # Top DRS performers
    print(f"\n🎯 DRS EFFECTIVENESS RANKING:")
    for i, driver_rank in enumerate(drs_predictions['effectiveness_ranking'][:5], 1):
        print(f"   {i}. {driver_rank['driver']}: {driver_rank['effectiveness']:.2f} ({driver_rank['primary_zone']})")
    
    # Key DRS battles
    print(f"\n⚔️ PREDICTED DRS BATTLES:")
    for battle in drs_predictions['key_battles'][:3]:
        print(f"   • {battle['battle']}: {battle['zone']} ({battle['laps']})")
        print(f"     → {battle['prediction']}")
    
    # DRS Timeline
    print(f"\n⏱️ DRS STRATEGIC TIMELINE:")
    timeline = drs_predictions['strategic_timeline']
    for phase, data in list(timeline.items())[:3]:  # Show first 3 phases
        print(f"   {data['phase']}: {data['drs_strategy']}")
    
    # Specific DRS strategies for top drivers
    print(f"\n🧠 VERSTAPPEN DRS STRATEGY:")
    if 'verstappen' in drs_predictions['driver_strategies']:
        vmax_strategy = drs_predictions['driver_strategies']['verstappen']
        print(f"   Effectiveness: {vmax_strategy['effectiveness_rating']:.2f}")
        print(f"   Primary Zone: {vmax_strategy['primary_zone']}")
        print(f"   Risk Level: {vmax_strategy['risk_assessment']}")
        for scenario in vmax_strategy['offensive_scenarios'][:2]:
            print(f"   • {scenario}")
    
    print(f"\n💡 PIASTRI DRS STRATEGY (Championship Leader):")
    if 'piastri' in drs_predictions['driver_strategies']:
        oscar_strategy = drs_predictions['driver_strategies']['piastri']
        print(f"   Effectiveness: {oscar_strategy['effectiveness_rating']:.2f}")
        print(f"   Strategy: Championship protection mode")
        for scenario in oscar_strategy['defensive_scenarios'][:2]:
            print(f"   • {scenario}")
    
    # Championship implications
    champ_impact = race_predictions['championship_impact']
    print(f"\n🏆 CHAMPIONSHIP IMPLICATIONS:")
    print("-" * 30)
    
    new_standings = champ_impact['new_standings'][:3]
    for pos, (driver, points) in enumerate(new_standings, 1):
        change = champ_impact['position_changes'][driver]
        pos_change = change['position_change']
        points_gained = change['points_gained']
        
        if pos_change > 0:
            change_str = f"(↑{pos_change})"
        elif pos_change < 0:
            change_str = f"(↓{abs(pos_change)})"
        else:
            change_str = "(=)"
        
        print(f"   {pos}. {driver.capitalize()}: {points} pts (+{points_gained}) {change_str}")
    
    gap_to_leader = champ_impact['title_fight']['leader_gap']
    print(f"\n📊 Title Fight: {gap_to_leader} points between P1 and P2")
    
    # Create visualization
    create_prediction_visualization(qualifying_predictions, race_predictions, drs_predictions, strategic_predictions)
    
    print(f"\n✅ 2025 AUSTRIAN GP PREDICTIONS COMPLETE!")
    print(f"🎯 AI Confidence Level: 85%")
    print(f"🏁 Key Factor: Verstappen home race performance")
    print(f"⚡ Wild Card: McLaren title fight pressure")
    print(f"🇦🇹 Don't miss the action: June 29, 2025!")
    
    return {
        'qualifying': qualifying_predictions,
        'race': race_predictions,
        'drs_strategy': drs_predictions,
        'strategic': strategic_predictions,
        'verstappen': verstappen_analysis
    }

if __name__ == "__main__":
    predictions = run_austria_gp_2025_predictions()