import os
import re
import csv

# Directory containing the text files
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# List of files to process
files = [
    'left_right.txt',
    'push_pull.txt',
    'push_no_moment.txt'
]

# Pattern to extract angle and force data
data_pattern = re.compile(r'angles: (\d+), (\d+).*?force: (\d+), (\d+)', re.DOTALL)

for filename in files:
    txt_path = os.path.join(data_dir, filename)
    csv_path = os.path.splitext(txt_path)[0] + '.csv'

    # Read the full file content
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all repetitive data entries in the format: angles: x, y ... force: a, b
    matches = data_pattern.findall(content)

    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['angle_1', 'angle_2', 'force_1', 'force_2'])
        for match in matches:
            writer.writerow(match)

    print(f"Extracted {len(matches)} entries from {filename} to {os.path.basename(csv_path)}")
