import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: API Key not found in .env")
else:
    print(f"🔑 Key found (ending in ...{api_key[-5:]})")
    genai.configure(api_key=api_key)

    print("\n🔍 ASKING GOOGLE FOR AVAILABLE MODELS...")
    try:
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✅ {m.name}")
                found_any = True
        
        if not found_any:
            print("   ⚠️ No chat models found. Check your API Key permissions.")
            
    except Exception as e:
        print(f"   ❌ Error connecting to Google: {e}")