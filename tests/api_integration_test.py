import requests
import time
import sys

BASE_URL = "http://localhost:8009"

def run_tests():
    print(f"🧪 connecting to {BASE_URL}...")
    
    # 1. Health Check
    try:
        r = requests.get(f"{BASE_URL}/health")
        if r.status_code == 200:
            print("✅ Health Check Passed")
        else:
            print(f"❌ Health Check Failed: {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        sys.exit(1)

    # 2. Search Check
    r = requests.get(f"{BASE_URL}/search", params={"q": "robot"})
    if r.status_code == 200:
        print("✅ Search Endpoint Passed")
    else:
        print(f"❌ Search Endpoint Failed: {r.status_code}")
        sys.exit(1)

    # 3. Bad Request Check (Error Handling 400)
    # Sending a text file should fail with 400 now
    bad_payload = {"path": f"{sys.path[0]}/tests/bad_file.txt", "force_reprocess": True}
    
    # Create dummy file if not exists
    with open("tests/bad_file.txt", "w") as f:
        f.write("I am not a video")
        
    r = requests.post(f"{BASE_URL}/process", json=bad_payload)
    if r.status_code == 400:
        print(f"✅ Error Handling Passed (Got expected 400: {r.json().get('detail')})")
    else:
        print(f"❌ Error Handling Failed! Expected 400, got {r.status_code}")
        print(f"   Response: {r.text}")
        sys.exit(1)

    print("🎉 All API Tests Passed!")

if __name__ == "__main__":
    # Wait for server to boot
    time.sleep(2)
    run_tests()
