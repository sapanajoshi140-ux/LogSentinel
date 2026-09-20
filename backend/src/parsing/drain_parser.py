# simple one-pass parser (baseline//sanity checks)
"""
Parse raw log files into structured (template_id, template, params, raw_line)
records using Drain3.

Usage:
    python -m src.parsing.drain_parser --input data/raw/HDFS.log \
        --output data/processed/HDFS_parsed.csv [--sample 1000]

This is intentionally dataset-agnostic at this stage: it treats each line
as free text and lets Drain3 mine templates. Dataset-specific fields
(block IDs for HDFS, node IDs for BGL) get extracted in src/splitting/,
once we know the raw lines parse cleanly.
"""

import argparse     # for giving input/output from cmd
import csv          # for creating the csv output
from pathlib import Path        #For handling the paths

from drain3 import TemplateMiner        # main drain3 parser
from drain3.template_miner_config import TemplateMinerConfig        #configures the drain3
from tqdm import tqdm       #shows the progress bar


# To create and return drain3 parser
def build_template_miner() -> TemplateMiner:
    """Create a Drain3 TemplateMiner with sane defaults for system logs."""
    config = TemplateMinerConfig()          #creates the default Drain3 configuration
    config.profiling_enabled = False        # as we dont need performance profilinng for this parser we disabled the profiling
    return TemplateMiner(config=config)     # creates actual Drain3 TemplateMiner


#The main processing function
def parse_file(input_path: Path, output_path: Path, sample: int | None = None) -> None:       #three inputs needed input raw log file, output path where parsed csv should be saved, and number of line to process from the log file 
    miner = build_template_miner()      #this creates the Drain3 parser

    with input_path.open("r", errors="ignore") as infile:
        lines = infile.readlines()       #reads all the lines in the input log file

    if sample:
        lines = lines[:sample]    #for processing n samples inputted

    output_path.parent.mkdir(parents=True, exist_ok=True)  #creates the output directory

    with output_path.open("w", newline="") as outfile:    #opens the csv file in write mode
        writer = csv.writer(outfile)    #creates the csv writer
        writer.writerow(["line_no", "template_id", "template", "raw_line"])     #creates the csv headers/columns -- there are four columns

        for i, line in enumerate(tqdm(lines, desc=f"Parsing {input_path.name}")):       #loop through every log line, tqdm provides progress bar
            line = line.strip()     #remove unnecessary whitespace
            if not line:            #if line is empty skip it
                continue
            result = miner.add_log_message(line)            #send the line to Drain3
            writer.writerow([i, result["cluster_id"], result["template_mined"], line]) #write the result to csv

    n_templates = len(miner.drain.clusters)     #counts how many templates Drain3 learned
    print(f"\nParsed {len(lines)} lines into {n_templates} templates.")         
    print(f"Output written to {output_path}")       #output path shown

    print("\nTop 10 most frequent templates:")  #shows top 10 templates 
    clusters = sorted(miner.drain.clusters, key=lambda c: c.size, reverse=True)     # sort templates acc to how many log message belongs to them
    for c in clusters[:10]:                     #takes top 10 templates
        print(f"  [{c.size:>6}]  {c.get_template()}")


# This handles the command-line arguments.
def main() -> None:
    parser = argparse.ArgumentParser(description="Parse logs with Drain3")              
    parser.add_argument("--input", required=True, type=Path, help="Path to raw log file")       #to get input file
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output CSV (defaults to data/processed/<input_stem>_parsed.csv)",
    )                                                   #output path is optional, if not provided creates the default
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only parse the first N lines (useful for a quick sanity check)",
    )           #samples to be processed
    args = parser.parse_args()                  #to read what user typed in terminal

    output = args.output or Path("data/processed") / f"{args.input.stem}_parsed.csv"
    parse_file(args.input, output, args.sample)         #actual parsing starts 


if __name__ == "__main__":
    main()
