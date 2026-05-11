import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_chat():
    messages = [
        {"role": "user", "content": "I am hiring a Java developer"}
    ]
    
    print("\n--- Turn 1 (Vague Query) ---")
    response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    res_json = response.json()
    print(json.dumps(res_json, indent=2))
    
    # Simulate user answering the agent's clarification
    messages.append({"role": "assistant", "content": res_json.get("reply", "")})
    messages.append({"role": "user", "content": "Mid-level, around 4 years. I need to assess their knowledge of Java 8 and MVC."})
    
    print("\n--- Turn 2 (Refinement) ---")
    response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    res_json = response.json()
    print(json.dumps(res_json, indent=2))
    
    messages.append({"role": "assistant", "content": res_json.get("reply", "")})
    messages.append({"role": "user", "content": "Just give me the assessments you have for Java 8 and MVC."})
    
    print("\n--- Turn 3 (Recommendation) ---")
    response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
    res_json = response.json()
    print(json.dumps(res_json, indent=2))

if __name__ == "__main__":
    test_health()
    test_chat()
