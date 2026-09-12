import pandas as pd
import numpy as np

class EWMADetector:
    def __init__(self, alpha=0.3, threshold_std=1.5):
        """
        alpha: Smoothing factor (0 < alpha <= 1) as defined in Section 4.3.3.
        threshold_std: Deviation multiplier to mark anomalies.
        """
        self.alpha = alpha
        self.threshold_std = threshold_std
        self.ewma_baseline = 0
        self.std_dev = 0

    def fit(self, sequence_df):
        """Calculates historical EWMA and std deviation for template/sequence lengths."""
        lengths = sequence_df['event_sequence'].apply(lambda x: len(eval(x) if isinstance(x, str) else x)).values
        
        # Section 4.3.3 Formula implementation: EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
        ewma_values = [lengths[0]]
        for t in range(1, len(lengths)):
            new_ewma = self.alpha * lengths[t] + (1 - self.alpha) * ewma_values[-1]
            ewma_values.append(new_ewma)
            
        self.ewma_baseline = ewma_values[-1]
        self.std_dev = np.std(lengths)
        print(f"[EWMA Detector Fitted] Final EWMA Baseline: {self.ewma_baseline:.2f}, Std Dev: {self.std_dev:.2f}")

    def predict(self, sequence_df):
        """Flags sequences whose length deviates significantly from EWMA baseline."""
        df = sequence_df.copy()
        df['seq_len'] = df['event_sequence'].apply(lambda x: len(eval(x) if isinstance(x, str) else x))
        
        # Calculate deviation from EWMA
        upper_bound = self.ewma_baseline + (self.threshold_std * self.std_dev)
        lower_bound = self.ewma_baseline - (self.threshold_std * self.std_dev)
        
        # Anomaly Flag (1 = Anomaly, 0 = Normal)
        df['is_anomaly'] = df['seq_len'].apply(lambda l: 1 if (l > upper_bound or l < lower_bound) else 0)
        return df