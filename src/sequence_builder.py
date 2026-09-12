import re
import pandas as pd

class SequenceBuilder:
    def __init__(self, block_regex=r'(blk_[-?\d]+)'):
        self.block_regex = re.compile(block_regex)

    def extract_block_id(self, log_message):
        """Log message mein se block_id extract karta hai."""
        match = self.block_regex.search(str(log_message))
        return match.group(1) if match else None

    def create_sequences(self, df):
        """Parsed log dataframe ko block_id ke hisab se group karta hai."""
        df = df.copy()
        
        # Block ID extract karein
        df['block_id'] = df['template'].apply(self.extract_block_id)
        
        # Null values remove karein
        df_clean = df.dropna(subset=['block_id'])
        
        # Groupby block_id
        sequences = df_clean.groupby('block_id')['template_id'].apply(list).reset_index()
        sequences.columns = ['block_id', 'event_sequence']
        
        return sequences