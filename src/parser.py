import os
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

class LogParser:
    def __init__(self, config_path="config/drain3.ini"):
        config = TemplateMinerConfig()
        if os.path.exists(config_path):
            config.load(config_path)
        self.template_miner = TemplateMiner(config=config)

    def parse_log_line(self, log_line):
        result = self.template_miner.add_log_message(log_line.strip())
        return {
            "template_id": str(result["cluster_id"]),
            "template": result["template_mined"]
        }

    def parse_log_file(self, file_path, max_lines=1000):
        parsed_records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                parsed = self.parse_log_line(line)
                parsed_records.append(parsed)
        return pd.DataFrame(parsed_records)