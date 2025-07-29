# goal is to filter a collection of CC WET files to produce language
# modeling training data. We’ve placed 5000 WET files at /data/CC/CC*.warc.wet.gz for you to use as a
# starting point.

import numpy as np
from transformers import AutoTokenizer
import concurrent.futures
import os
from tqdm import tqdm
import pathlib

# TODO: no access to this data
data = np.fromfile("/data/paloma/tokenized_paloma_c4_100_domains_validation.bin", dtype=np.uint16)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer.decode(data[0:2000]))

# allowed to make use of paloma validation data in constructing filters or classifiers to process the cc wet files
# fuck, 375 GB of data required :/
# gotta do it probs on GCP, and ssh there


def process_single_wet_file(input_path: str, output_path: str):
    # TODO: read input path, process the input, and write the output to output_path
    return output_path


# Set up the executor
num_cpus = len(os.sched_getaffinity(0))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
output_directory_path = "/path/to/output_directory/"
futures = []
for wet_filepath in wet_filepaths:
    # For each warc.wet.gz filepath, submit a job to the executor and get a future back
    wet_filename = str(pathlib.Path(wet_filepath).name)
    future = executor.submit(process_single_wet_file, wet_filepath, os.path.join(output_directory_path, wet_filename))
    # Store the futures
    futures.append(future)
# Iterate over the completed futures as they finish, using a progress bar
# to keep track of progress.
for future in tqdm(
    concurrent.futures.as_completed(futures),
    total=len(wet_filepaths),
):
    output_file = future.result()
    print(f"Output file written: {output_file}")
