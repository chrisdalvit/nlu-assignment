import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)

# Script to preprocess the dataset into the right format
# This allows the reusage of code from previous tasks

if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(args.path) as file:
        data = []
        for line in file.readlines():
            # Split line into sentence and token to slot mapping
            utt, slot_map = line.split("####")
            
            tokens = []
            slots = []
            # For every token to slot mapping, collect tokens and slots
            for slot in slot_map.split():
                parts = slot.rsplit("=", maxsplit=1)
                if len(parts) == 2:
                    token, slot = parts
                    tokens.append(token)
                    slots.append(slot)
                else:
                    print(f"Error on {utt} with {slot_map}")
            data.append({
                "sentence": utt,
                "tokens": tokens,
                "slots": slots
            })

    # Write the processed data into a new file
    filename = os.path.basename(args.path)
    if not os.path.exists("dataset_processed"):
        os.mkdir("dataset_processed")
    with open(f"dataset_processed/{filename.split('.')[0]}.json", mode="w") as file:
        output = json.dumps(data)
        file.write(output)      