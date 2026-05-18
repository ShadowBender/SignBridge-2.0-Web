import os
import google.generativeai as genai

# 1. Manually grab the key just like app.py does
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#") and "=" in cleaned:
                k, v = cleaned.split("=", 1)
                os.environ[k.strip()] = v.strip()

api_key = os.environ.get("GEMINI_API_KEY", "")

print(f"--- DIAGNOSTIC TEST ---")
print(f"1. API Key Loaded: {api_key[:10]}... (Hidden for security)")

# 2. Configure Google
genai.configure(api_key=api_key)

# We are using the most universally stable model version for this test
model = genai.GenerativeModel('gemini-1.5-flash')

print("2. Contacting Google Servers...")

try:
    response = model.generate_content("Translate this broken sign language into perfect English: How You")
    print(f"\n✅ SUCCESS! Google's brain is working.")
    print(f"AI Translation: '{response.text.strip()}'\n")
except Exception as e:
    print(f"\n❌ CRITICAL ERROR! Google rejected the request.")
    print(f"Exact Reason: {e}\n")