import requests
import dotenv
import hashlib
import os

dotenv.load_dotenv()

def is_file_safe(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        raise ValueError("Invalid file path")

    # Get file hash
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

 

    # Check if file exists on VirusTotal
    headers = {"x-apikey": os.getenv("VIRUSTOTAL_API_KEY")}
    response = requests.get(f"https://www.virustotal.com/api/v3/files/{file_hash}", headers=headers)

    if response.status_code == 200:
        data = response.json()
        malicious_votes = data["data"]["attributes"]["last_analysis_stats"]["malicious"]
        return malicious_votes == 0  # True if clean
    else:
        # Upload file if not found
        files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
        upload_response = requests.post("https://www.virustotal.com/api/v3/files", headers=headers, files=files)
        if upload_response.status_code == 200:
            print("File uploaded. Scan in progress. Try again later.")
            return False
        else:
            print("Error uploading file to VirusTotal.")
            return False
if __name__ == "__main__":
    path = input("Enter path to file: ")
    safe = is_file_safe(path)
    print("✅ File is clean." if safe else "⚠️ File may be malicious.")

