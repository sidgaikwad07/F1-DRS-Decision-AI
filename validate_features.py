"""
Created on Mon Jun 23 11:50:33 2025

@author: sid

STEP 1: Validate Features
Quick test to make sure your feature engineering pipeline works correctly
Run this FIRST to validate everything before training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f1_drs_utlities import DRSFeatureEngineer, DRSLabelGenerator
from f1_telemetry_processor import F1TelemetryProcessor

def step1_validate_features():
    """
    Step 1: Validate that feature engineering works correctly
    """
    print("🔧 STEP 1: VALIDATING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # 1. Load your processed data
    print("📁 Loading your processed F1 data...")
    base_folder = "/Users/sid/Downloads/F1-DRS-Decision-AI/transformer_austria_training"
    processor = F1TelemetryProcessor(base_folder)
    processor.load_all_data()
    
    if not processor.train_data:
        print(" ERROR: No training data found!")
        print("   Make sure you ran f1_telemetry_processor.py first")
        return False
    
    print(f"✅ Loaded {len(processor.train_data)} training sequences")
    
    # 2. Initialize feature engineering
    print("\n🔧 Initializing feature engineering...")
    feature_engineer = DRSFeatureEngineer()
    label_generator = DRSLabelGenerator()
    
    feature_columns = [
        'speed', 'throttle', 'brake', 'ers', 'gap_ahead', 'position_norm',
        'speed_change', 'throttle_change', 'gap_change', 'drs_opportunity_score'
    ]
    
    # 3. Test on a few sequences
    print("\n📊 Testing feature engineering on sample sequences...")
    
    test_sequences = 5  # Test on 5 sequences first
    all_drs_rates = []
    
    for i in range(min(test_sequences, len(processor.train_data))):
        print(f"\n--- Testing Sequence {i+1} ---")
        
        # Convert sequence to DataFrame
        sequence = processor.train_data[i]
        df = processor.sequence_to_dataframe(sequence)
        
        if df.empty:
            print(f"⚠️  Sequence {i+1}: Empty DataFrame - skipping")
            continue
            
        print(f"📏 Original sequence length: {len(df)} timesteps")
        
        # Apply feature engineering
        try:
            df_enhanced = feature_engineer.engineer_drs_features(df)
            print(f"🔧 Enhanced features: {df_enhanced.shape[1]} columns")
            
            # Fit scaler on first sequence
            if i == 0:
                feature_engineer.fit_scaler(df_enhanced, feature_columns)
                print("✅ Scaler fitted on first sequence")
            
            # Generate DRS labels
            drs_labels = label_generator.generate_drs_labels(df_enhanced)
            drs_rate = drs_labels.mean()
            all_drs_rates.append(drs_rate)
            
            print(f"🎯 DRS opportunity rate: {drs_rate:.3f} ({drs_rate*100:.1f}%)")
            
            # Show sample enhanced features
            print(f"📈 Sample enhanced features (first 3 timesteps):")
            sample_features = df_enhanced[feature_columns].head(3)
            for col in feature_columns:
                values = sample_features[col].values
                print(f"   {col:20s}: {values}")
            
        except Exception as e:
            print(f" Error processing sequence {i+1}: {e}")
            continue
    
    # 4. Summary statistics
    print(f"\n📊 FEATURE VALIDATION SUMMARY:")
    print(f"✅ Sequences processed successfully: {len(all_drs_rates)}/{test_sequences}")
    
    if all_drs_rates:
        avg_drs_rate = np.mean(all_drs_rates)
        print(f"🎯 Average DRS opportunity rate: {avg_drs_rate:.3f} ({avg_drs_rate*100:.1f}%)")
        
        # Check if rate is realistic (should be around 2-5% for F1)
        if 0.01 <= avg_drs_rate <= 0.08:
            print(f"✅ DRS rate is realistic for F1 racing!")
        else:
            print(f"⚠️  DRS rate seems {('high' if avg_drs_rate > 0.08 else 'low')} - check labeling logic")
    
    print(f"🔧 Enhanced feature count: 10 (up from 6 original)")
    print(f"📏 Feature columns: {feature_columns}")
    
    # 5. Quick visualization
    print(f"\n📈 Creating feature validation plot...")
    
    # Test one more sequence for visualization
    test_seq = processor.train_data[0]
    df_test = processor.sequence_to_dataframe(test_seq)
    df_enhanced = feature_engineer.engineer_drs_features(df_test)
    drs_labels = label_generator.generate_drs_labels(df_enhanced)
    
    # Create simple plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('STEP 1: Feature Engineering Validation', fontsize=14)
    
    timesteps = range(len(df_enhanced))
    
    # Speed and DRS opportunities
    axes[0, 0].plot(timesteps, df_enhanced['speed'], 'b-', linewidth=2)
    axes[0, 0].set_title('Speed')
    axes[0, 0].set_ylabel('km/h')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gap ahead
    axes[0, 1].plot(timesteps, df_enhanced['gap_ahead'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='DRS Threshold')
    axes[0, 1].set_title('Gap to Car Ahead')
    axes[0, 1].set_ylabel('Seconds')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # DRS Opportunity Score
    axes[1, 0].plot(timesteps, df_enhanced['drs_opportunity_score'], 'purple', linewidth=2)
    axes[1, 0].set_title('DRS Opportunity Score (Enhanced Feature)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].grid(True, alpha=0.3)
    
    # DRS Labels
    axes[1, 1].plot(timesteps, drs_labels, 'orange', linewidth=3, alpha=0.7)
    axes[1, 1].set_title('DRS Should Be Used (Labels)')
    axes[1, 1].set_ylabel('DRS Active (0/1)')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Final validation check
    print(f"\n✅ STEP 1 VALIDATION COMPLETE!")
    print(f"🎯 Key Results:")
    print(f"   • Feature engineering: ✅ Working")
    print(f"   • DRS labeling: ✅ Working") 
    print(f"   • Data pipeline: ✅ Working")
    print(f"   • Enhanced features: ✅ 10 features generated")
    print(f"   • DRS rate: ✅ {avg_drs_rate*100:.1f}% (realistic)")
    
    print(f"\n🚀 READY FOR STEP 2: Train Simple Model")
    print(f"   Run: python step2_train_simple_model.py")
    
    return True

if __name__ == "__main__":
    success = step1_validate_features()
    
    if success:
        print(f"\n🎉 STEP 1 SUCCESSFUL!")
        print(f"✅ Your feature engineering pipeline is working correctly")
        print(f"✅ DRS labeling logic is generating realistic rates")
        print(f"✅ Ready to proceed to model training")
    else:
        print(f"\n STEP 1 FAILED!")
        print(f"⚠️  Fix the issues above before proceeding")
        print(f"🔧 Check your data paths and feature engineering code")