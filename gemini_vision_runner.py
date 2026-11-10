import google.generativeai as genai
import os
import json
from PIL import Image

# Initialize the model: Gemini 1.5 Pro for maximum accuracy on Footprint Order Flow charts
# Ensure you have your GEMINI_API_KEY set in your environment variables
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel('gemini-1.5-pro')

def analyze_footprint_chart(image_path):
    """
    Takes a path to a footprint chart image, passes it to Gemini 1.5 Pro, 
    and strictly enforces extraction to a predefined JSON format.
    """
    
    # 1. Load the chart image
    try:
        chart_image = Image.open(image_path)
    except FileNotFoundError:
        return {"analysis_failed": True, "error": "Chart image not found"}

    # 2. Define the strict instruction prompt
    system_prompt = """
    You are an expert quantitative trading vision model. Analyze the provided order flow footprint chart.
    
    Extract the following context and strictly output ONLY valid JSON format. Do not include markdown formatting or conversational text.
    
    1. Check for specific textual context notes (e.g. "Aggressive Buying", "RTH Open above ETH VAH").
    2. Identify if there is a clear visual pattern indicating "Buyers Absorb Sellers" at key Support/Resistance nodes.
    3. Output your reading into the following strict JSON schema:
    
    {
      "signal_at": "YYYY-MM-DDTHH:MM:SSZ",
      "no_trade": [],
      "analysis_failed": false,
      "trades": [
        {
          "symbol": "NQ-202612",
          "action": "BUY",  # or "SELL" based on absorption
          "entry": 0.0,     # Extract or estimate key level
          "stop_loss": 0.0, 
          "take_profit": 0.0,
          "rr": 0.0,
          "digits": 2,
          "context_notes": ["Aggressive Buying", "RTH Open..."]
        }
      ]
    }
    
    If the chart is unreadable or the pattern is missing, set "analysis_failed": true or add reasons to "no_trade".
    """

    # 3. Call the Pro model for maximum accuracy
    try:
        response = model.generate_content([system_prompt, chart_image])
        
        # Clean the output in case the model wraps it in markdown blocks
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        
        # Parse it to ensure it is valid JSON before returning
        return json.loads(clean_json)
        
    except Exception as e:
        return {"analysis_failed": True, "error": str(e)}

# --- Execution & Logging ---
if __name__ == "__main__":
    # 1. Analyze the chart
    result = analyze_footprint_chart("latest_chart.png")
    
    # 2. Update the live dashboard instantly
    with open('latest_signal.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    # 3. Append to the running log (history.json)
    history = []
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass # Start fresh if file is empty/corrupted
                
    history.append(result)
    with open('history.json', 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"Signal logged. Analysis failed: {result.get('analysis_failed')}")
