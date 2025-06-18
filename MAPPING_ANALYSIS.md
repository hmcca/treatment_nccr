# Regimen-Drug Mapping Analysis

## Current Behavior Analysis

After examining the code and data, I found that the `mapped_regimen_drugs` logic is working correctly. Here's what's happening:

### Column Definitions

1. **`mapped_regimen_drugs`**: JSON string showing the mapping structure between regimens and their constituent drugs
   - Example: `{"chop": ["vincristine", "prednisone", "dapsone", "granulocyte colony-stimulating factor", ...]}`
   - This shows **which regimen maps to which drugs**

2. **`combined_drugs`**: Flat list of all unique drugs from both extracted drugs and regimen mappings
   - Example: `["granulocyte colony-stimulating factor", "caplacizumab", "filgrastim", "vincristine", ...]`
   - This shows **all drugs combined** from both sources

### Case Sensitivity Analysis

The case sensitivity is handled correctly:
- Regimen names in the mapping file (e.g., "CHOP") are converted to lowercase when loaded
- Extracted regimens (e.g., "chop") are also converted to lowercase
- The matching works correctly between "chop" and "chop"

### Current Mapping Examples

From the checkpoint data:

**Example 1: CHOP Regimen**
- Extracted drugs: `['caplacizumab', 'rituximab']`
- Extracted regimens: `['chop']`
- Combined drugs: `['granulocyte colony-stimulating factor', 'caplacizumab', 'filgrastim', 'vincristine', 'trimethoprim-sulfamethoxazole', 'prednisone', 'doxorubicin', 'rituximab', 'cyclophosphamide', 'dapsone', 'pentamidine']`
- Mapped regimen drugs: `{"chop": ["vincristine", "prednisone", "dapsone", "granulocyte colony-stimulating factor", "pentamidine", "trimethoprim-sulfamethoxazole", "doxorubicin", "filgrastim", "cyclophosphamide"]}`

**Example 2: No Regimens**
- Extracted drugs: `['doxorubicin', 'prednisone', 'atezolizumab']`
- Extracted regimens: `[]`
- Combined drugs: `['doxorubicin', 'prednisone', 'atezolizumab']`
- Mapped regimen drugs: `{}`

## Issues Identified

### 1. Potential Confusion About Column Purpose
The user might be confused about the difference between:
- `mapped_regimen_drugs` (JSON structure showing mappings)
- `combined_drugs` (flat list of all drugs)

### 2. Limited Regimen Extraction
Most records only extract "chop" as a regimen, suggesting the LLM might not be identifying other regimens properly.

### 3. Missing Additional Columns
The current implementation doesn't provide a clear way to see:
- Drugs that come only from regimens (not directly extracted)
- Drugs that come only from direct extraction (not from regimens)

## Recommended Improvements

### 1. Add Clearer Column Names and Documentation

```python
# Current columns
'mapped_regimen_drugs'  # JSON showing regimen->drug mappings
'combined_drugs'        # All unique drugs from both sources

# Suggested additional columns
'regimen_drugs_only'    # Drugs that come only from regimen mappings
'extracted_drugs_only'  # Drugs that come only from direct extraction
'regimen_drugs_flat'    # Flat list of drugs from regimens only
```

### 2. Improve Logging and Debugging

The improved version includes better logging to show:
- Which regimens are being matched
- How many drugs each regimen maps to
- Sample mappings for verification

### 3. Enhanced Regimen Matching

Consider improving the regimen extraction by:
- Adding more regimen patterns to the LLM prompt
- Implementing post-processing to identify common regimen patterns
- Adding validation for regimen names

### 4. Better Data Validation

Add validation to ensure:
- Regimen names are properly normalized
- Drug names are consistent
- Mappings are complete

## Implementation Status

The mapping logic is working correctly. The main improvements needed are:

1. **Better documentation** of what each column contains
2. **Additional columns** to provide different views of the data
3. **Enhanced logging** for debugging and verification
4. **Improved regimen extraction** to capture more regimens

## Files Created

1. `drug_normalizer_improved.py` - Enhanced version with better logging
2. `main_improved.py` - Improved main processing with additional columns
3. `test_mapping.py` - Test script to demonstrate the mapping behavior

## Next Steps

1. Review the improved implementation
2. Test with a small dataset to verify the behavior
3. Consider adding more regimen patterns to improve extraction
4. Add validation and error handling for edge cases

The core mapping logic is sound - the issue appears to be more about clarity and additional functionality rather than bugs in the existing code. 