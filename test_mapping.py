#!/usr/bin/env python3
"""
Test script to demonstrate the regimen-drug mapping behavior
"""

import pandas as pd
from drug_normalizer import RegimenDrugMapper

def test_mapping():
    """Test the regimen-drug mapping functionality"""
    
    # Initialize the mapper
    mapper = RegimenDrugMapper('regimen_drug_mapping.csv')
    
    # Test cases
    test_cases = [
        {
            'extracted_drugs': ['caplacizumab', 'rituximab'],
            'extracted_regimens': ['chop'],
            'description': 'CHOP regimen with additional drugs'
        },
        {
            'extracted_drugs': ['doxorubicin', 'prednisone', 'atezolizumab'],
            'extracted_regimens': [],
            'description': 'No regimens, only direct drugs'
        },
        {
            'extracted_drugs': ['cyclophosphamide', 'doxorubicin', 'vincristine'],
            'extracted_regimens': ['blinatumomab'],
            'description': 'Blinatumomab as regimen'
        }
    ]
    
    print("Testing Regimen-Drug Mapping Logic\n")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['description']}")
        print("-" * 40)
        
        extracted_drugs = case['extracted_drugs']
        extracted_regimens = case['extracted_regimens']
        
        print(f"Extracted Drugs: {extracted_drugs}")
        print(f"Extracted Regimens: {extracted_regimens}")
        
        # Get drugs from regimens using the improved method
        regimen_drugs_only = mapper.get_mapped_regimen_drugs_flat(extracted_regimens)
        
        for regimen in extracted_regimens:
            if regimen:
                drugs = mapper.get_drugs_from_regimen(regimen)
                print(f"  Regimen '{regimen}' -> Drugs: {drugs}")
        
        # Get combined drugs
        combined_drugs = mapper.get_combined_drugs(extracted_drugs, extracted_regimens)
        
        print(f"\nResults:")
        print(f"  Regimen Drugs Only (flat list): {regimen_drugs_only}")
        print(f"  Combined Drugs (flat list): {combined_drugs}")
        print(f"  Total unique drugs: {len(combined_drugs)}")

if __name__ == "__main__":
    test_mapping() 