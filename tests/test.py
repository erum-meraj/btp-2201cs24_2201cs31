import google.generativeai as genai
import os, dotenv

# Make sure to set your GOOGLE_API_KEY environment variable
# or configure it directly:
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

print("Available Google GenAI Models:")

# list_models() returns an iterator
for model in genai.list_models():
  # We're checking if 'generateContent' is a supported method
  # to filter for the chat/text models you likely want.
  if 'generateContent' in model.supported_generation_methods:
    print(f"- {model.name}")