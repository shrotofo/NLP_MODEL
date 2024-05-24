import json
import re

# Read the JSON file as a string
with open('nlp.json', 'r') as file:
    json_data = file.read()

# Insert commas between JSON objects
json_data = re.sub(r'}\s*{', '},{', json_data)

# Ensure the JSON data is wrapped in an array
if not json_data.startswith('['):
    json_data = '[' + json_data
if not json_data.endswith(']'):
    json_data = json_data + ']'

# Validate the JSON data
try:
    parsed_json = json.loads(json_data)
    print("JSON data has been successfully fixed and validated.")
    
    # Save the corrected JSON data to a new file
    with open('corrected_data.json', 'w') as corrected_file:
        json.dump(parsed_json, corrected_file, indent=4)
    print("Corrected JSON data has been saved to 'corrected_data.json'.")
except json.JSONDecodeError as e:
    print("Failed to parse JSON data:", e)
