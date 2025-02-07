import sys
import json
import re
from transformers import pipeline

def correct_text(text):
    # Load a pre-trained NLP model for text correction
    corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")

    # Generate corrected text
    corrected_text = corrector(text, max_length=512, truncation=True)[0]['generated_text']

    # Capitalize the first letter of each sentence and remove any leading/trailing whitespace
    formatted_text = re.sub(r'(^\s*|\.\s+)([a-z])', lambda p: p.group(1) + p.group(2).upper(), corrected_text.strip())

    return formatted_text

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input text provided"}))
        return

    input_text = sys.argv[1]

    try:
        corrected_text = correct_text(input_text)
        output = {"correctedText": corrected_text}
    except Exception as e:
        output = {"error": f"Error processing text: {str(e)}"}

    # Print JSON response to be sent back to Node.js
    print(json.dumps(output))

if __name__ == "__main__":
    main()
