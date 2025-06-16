import json
import os
import requests
from tqdm import tqdm

subdir = "data"
os.makedirs(subdir, exist_ok=True)


def download_datasets():
    for ds in [
        # "webtext",
        # "small-117M",
        "small-117M-k40",
        # "medium-345M",
        # "medium-345M-k40",
        # "large-762M",
        # "large-762M-k40",
        # "xl-1542M",
        # "xl-1542M-k40",
    ]:
        for split in ["train", "valid", "test"]:
            filename = ds + "." + split + ".jsonl"
            r = requests.get(
                "https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"
                + filename,
                stream=True,
            )

            with open(os.path.join(subdir, filename), "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(
                    ncols=100,
                    desc="Fetching " + filename,
                    total=file_size,
                    unit_scale=True,
                ) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

def explore_dataset():
    with open("data/small-117M.test.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
        print(f"Loaded {len(data)} examples")
        print("First example:", data[0])
    
    longest = max([i["length"] for i in data])
    print("longest sequence", longest)

if __name__ == "__main__":
    # download_datasets()
    explore_dataset()
