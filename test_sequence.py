import pandas as pd
from src.sequence_builder import SequenceBuilder


input_csv = "data/processed/parsed_templates.csv"
print(f"Loading parsed logs from {input_csv}...")
df = pd.read_csv(input_csv)


builder = SequenceBuilder()
sequence_df = builder.create_sequences(df)


print("\n--- Extracted Block Sequences ---")
print(sequence_df.head())


output_csv = "data/processed/block_sequences.csv"
sequence_df.to_csv(output_csv, index=False)
print(f"\nSuccess! Block sequences saved to {output_csv}")
