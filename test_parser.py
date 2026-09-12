import os
from src.parser import LogParser

# Parser initialize karein
parser = LogParser(config_path="config/drain3.ini")

# Sample log file path
sample_file_path = "data/raw/sample.log"

if not os.path.exists(sample_file_path) and os.path.exists("sample.log"):
    sample_file_path = "sample.log"

print(f"Parsing logs from: {sample_file_path}...")
df = parser.parse_log_file(sample_file_path, max_lines=100)

print("\n--- Parsed Logs Output ---")
print(df.head())

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/parsed_templates.csv", index=False)
print("\nSuccess! Parsed output saved to data/processed/parsed_templates.csv")