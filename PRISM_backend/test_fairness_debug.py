"""
Test script to debug fairness metrics feature detection
"""
import numpy as np
import pandas as pd

def test_sensitive_feature_detection():
    """Test the sensitive feature detection logic"""
    
    # Define sensitive features (same as in ml_service.py)
    sensitive_features = [
        # Demographics  
        "gender", "sex", "age", "race", "ethnicity", "religion", "nationality", "citizenship",  
        "disability", "disability_status", "pregnancy_status",  

        # Socioeconomic & Employment  
        "income", "salary", "wage", "education", "education_level", "marital_status",  
        "employment", "employment_status", "occupation", "job_title",  

        # Health & Genetic Information  
        "health", "health_status", "medical_history", "genetic_data", "disorder", "disease",  

        # Location & Citizenship  
        "birthplace", "country_of_birth", "residence", "zip_code", "postal_code",  

        # Other Identifiers  
        "criminal_record", "conviction_history", "political_affiliation", "sexual_orientation"
    ]

    # Test cases with different feature sets
    test_cases = [
        {
            "name": "Dataset with direct matches",
            "feature_names": ["feature1", "age", "income", "feature4"],
            "expected": ["age", "income"]
        },
        {
            "name": "Dataset with case mismatch",
            "feature_names": ["feature1", "Age", "INCOME", "feature4"],
            "expected": ["Age", "INCOME"]
        },
        {
            "name": "Dataset with partial matches",
            "feature_names": ["feature1", "age_group", "annual_income", "feature4"],
            "expected": ["age_group", "annual_income"]
        },
        {
            "name": "Dataset with no matches (categorical detection)",
            "feature_names": ["feature1", "feature2", "feature3", "feature4"],
            "expected": []  # Will depend on synthetic data
        },
        {
            "name": "Empty dataset",
            "feature_names": [],
            "expected": []
        }
    ]

    for test_case in test_cases:
        print(f"\n=== {test_case['name']} ===")
        feature_names = test_case["feature_names"]
        print(f"Input features: {feature_names}")
        
        # Apply the same logic as in ml_service.py
        valid_sensitive_features = []
        
        # Strategy 1: Exact matches (case-sensitive)
        available_features = set(feature_names)
        exact_matches = [f for f in sensitive_features if f in available_features]
        valid_sensitive_features.extend(exact_matches)
        if exact_matches:
            print(f"Found exact matches: {exact_matches}")
        
        # Strategy 2: Case-insensitive matches
        if not valid_sensitive_features:
            print("No exact matches found, trying case-insensitive matches")
            case_insensitive_matches = [
                f for f in feature_names 
                if any(f.lower() == s.lower() for s in sensitive_features)
            ]
            valid_sensitive_features.extend(case_insensitive_matches)
            if case_insensitive_matches:
                print(f"Found case-insensitive matches: {case_insensitive_matches}")
        
        # Strategy 3: Partial matches (contains)
        if not valid_sensitive_features:
            print("No case-insensitive matches found, trying partial matches")
            partial_matches = [
                f for f in feature_names 
                if any(s.lower() in f.lower() or f.lower() in s.lower() for s in sensitive_features)
            ]
            valid_sensitive_features.extend(partial_matches)
            if partial_matches:
                print(f"Found partial matches: {partial_matches}")
        
        # Strategy 4: Auto-detect categorical features (simulated)
        if not valid_sensitive_features and feature_names:
            print("No direct matches found, checking for categorical features")
            # Simulate categorical detection - in real code this would check actual data
            # For demo purposes, let's say features with certain patterns are categorical
            categorical_features = []
            for feature_name in feature_names:
                # Simple heuristic for demo
                if len(feature_name) < 12 and not feature_name.startswith('feature'):
                    categorical_features.append(feature_name)
                    print(f"Detected potentially sensitive categorical feature: {feature_name}")
            
            valid_sensitive_features.extend(categorical_features[:5])  # Limit to first 5
            
        # Remove duplicates while preserving order
        valid_sensitive_features = list(dict.fromkeys(valid_sensitive_features))
        
        print(f"Final valid sensitive features: {valid_sensitive_features}")
        print(f"Expected: {test_case['expected']}")
        
        # For this test, we'll check if we got reasonable results
        if test_case['name'] == "Dataset with no matches (categorical detection)":
            if valid_sensitive_features:
                print("✓ Categorical detection worked")
            else:
                print("ℹ No categorical features detected (which may be correct)")
        else:
            if set(valid_sensitive_features) == set(test_case['expected']):
                print("✓ Test passed")
            else:
                print("✗ Test failed")

def test_fairness_metrics_structure():
    """Test the fairness metrics structure"""
    print("\n=== Testing Fairness Metrics Structure ===")
    
    # Simulate what should be created
    valid_sensitive_features = ["gender", "income_level"]
    
    fairness_metrics = {
        'sensitive_features': valid_sensitive_features,
        'metrics': {},
        'statistical_tests': {},
        'interpretation': {}
    }
    
    # For each feature, create the structure
    for feature in valid_sensitive_features:
        feature_metrics = {
            'demographic_parity': {},
            'equal_opportunity': {},
            'equalized_odds': {},
            'disparate_impact': {},
            'treatment_equality': {},
            'statistical_parity': {},
            'statistical_tests': {},
            'interpretation': {
                'demographic_parity_threshold': 0.1,
                'equal_opportunity_threshold': 0.1,
                'disparate_impact_threshold': 0.8,
                'statistical_parity_threshold': 0.1
            }
        }
        fairness_metrics['metrics'][feature] = feature_metrics
    
    print("Generated fairness metrics structure:")
    import json
    print(json.dumps(fairness_metrics, indent=2))
    
    # Check if it has the expected structure
    assert 'sensitive_features' in fairness_metrics
    assert 'metrics' in fairness_metrics
    assert len(fairness_metrics['metrics']) == len(valid_sensitive_features)
    
    for feature in valid_sensitive_features:
        assert feature in fairness_metrics['metrics']
        feature_data = fairness_metrics['metrics'][feature]
        expected_keys = [
            'demographic_parity', 'equal_opportunity', 'equalized_odds',
            'disparate_impact', 'treatment_equality', 'statistical_parity',
            'statistical_tests', 'interpretation'
        ]
        for key in expected_keys:
            assert key in feature_data
    
    print("✓ Fairness metrics structure test passed")

if __name__ == "__main__":
    test_sensitive_feature_detection()
    test_fairness_metrics_structure()
    print("\n=== All tests completed ===") 