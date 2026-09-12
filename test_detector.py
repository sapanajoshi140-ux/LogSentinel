import pandas as pd
from src.detector import EWMADetector

# 1. Load block sequences
input_csv = "data/processed/block_sequences.csv"
print(f"Loading block sequences from {input_csv}...")
df = pd.read_csv(input_csv)

# 2. Fit EWMA Detector
detector = EWMADetector(alpha=0.3, threshold_std=1.5)
detector.fit(df)

# 3. Predict Anomalies
results_df = detector.predict(df)

# 4. Summary
anomalies = results_df[results_df['is_anomaly'] == 1]
print(f"\n--- EWMA Detection Summary ---")
print(f"Total Blocks Analyzed: {len(results_df)}")
print(f"Anomalies Found: {len(anomalies)}")

if len(anomalies) > 0:
    print("\nAnomalous Blocks Details:")
    print(anomalies[['block_id', 'seq_len', 'is_anomaly']])
else:
    print("\nNo anomalies detected in the current block dataset.")

# 5. Save output
output_csv = "data/processed/anomaly_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nSuccess! Results saved to {output_csv}")