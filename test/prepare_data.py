import hashlib
import json


def sha256(text:str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
if __name__ == "__main__":
    with open("sample_emails.txt", "r") as file:
        text = file.read()
    list_text = text.split("--")
    print(len(list_text))
    for text in list_text:
        hashstr = sha256(text)
        path_file = f"test_data/{hashstr}.json"
        with open(path_file, "w") as file:
            json.dump({"text": text}, file, indent=4)
