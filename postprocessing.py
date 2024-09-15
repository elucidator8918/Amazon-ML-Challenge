import re
import pandas as pd
from entity_mapping import unit_variations


def extract_first_value(text):
    # Build a single regex pattern that includes all unit variations
    unit_pattern_parts = []
    unit_standardization = {}
    for standard_unit, variations in unit_variations.items():
        # Create a regex group for variations of the unit
        variations_escaped = [re.escape(var) for var in variations]
        unit_group = r'(' + '|'.join(variations_escaped) + r')'
        unit_pattern_parts.append(unit_group)
        
        # Map each variation to the standard unit
        for var in variations:
            unit_standardization[var.lower()] = standard_unit
    
    # Combine all unit groups into one pattern
    unit_pattern = '|'.join(unit_pattern_parts)
    
    # Pattern to match the number and unit
    pattern = rf'(\d+(?:\.\d+)?)(?:\s*(?:-|\sto\s)\s*\d+(?:\.\d+)?)?\s*(?:{unit_pattern})\b'
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        # Extract the value and unit
        value = match.group(1)
        unit = match.group(2)
        if not unit:
            # If unit is not captured in group 2, find which group matched
            for i, part in enumerate(unit_pattern_parts, start=2):
                if match.group(i):
                    unit = match.group(i)
                    break
        
        # Standardize the unit
        unit_standard = unit_standardization[unit.lower()]
        
        return f"{value} {unit_standard}"
    
    return None




data = pd.read_csv("test_out_qwen.csv")

data.dropna(subset=['prediction'], inplace=True)

data['extracted_value'] = data['prediction'].apply(extract_first_value)

data.to_csv("test_out_qwen_postprocessed.csv", index=False)



