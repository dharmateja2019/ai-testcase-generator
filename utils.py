import json
from datetime import datetime

def save_history(user_story, output):
    record = {
        "timestamp": str(datetime.now()),
        "user_story": user_story,
        "output": output
    }

    try:
        with open("history.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open("history.json", "w") as f:
        json.dump(data, f, indent=4)