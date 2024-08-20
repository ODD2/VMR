import json
import pickle
from tqdm import tqdm

with open("check_story.pickle", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print("Miss:", data["eval_miss_responses"])
print("Response:", set(data["eval_responses"]))
print("Count:", len(set(data["eval_meta_paths"])))

for r, p in tqdm(zip(data["eval_responses"], data["eval_meta_paths"]), total=len(data["eval_responses"])):
    try:
        with open(p, "r") as f:
            meta = json.load(f)

        meta["is_story"] = r.lower() == "yes"

        with open(p, "w") as f:
            json.dump(meta, f)

    except Exception as e:
        print(f"Path: {p} find error {e}")
