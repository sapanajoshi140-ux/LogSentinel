import pandas as pd
from src.sequence_builder import SequenceBuilder

# 1. Parsed templates CSV load karein
input_csv = "data/processed/parsed_templates.csv"
print(f"Loading parsed logs from {input_csv}...")
df = pd.read_csv(input_csv)

# 2. Sequence Builder run karein
builder = SequenceBuilder()
sequence_df = builder.create_sequences(df)

# 3. Output print karein
print("\n--- Extracted Block Sequences ---")
print(sequence_df.head())

# 4. Save to processed folder
output_csv = "data/processed/block_sequences.csv"
sequence_df.to_csv(output_csv, index=False)
print(f"\nSuccess! Block sequences saved to {output_csv}")