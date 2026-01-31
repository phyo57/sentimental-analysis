import os
import json
import torch
import re
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Azure AI Content Safety Imports
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions

def init():
    global model, tokenizer, device, safety_client
    
    # --- A. Load BERT Model ---
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "my-final-bert-model")
    logging.info(f"Loading model from: {model_path}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise e

    # --- B. Load Content Safety Client ---
    endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT")
    key = os.environ.get("CONTENT_SAFETY_KEY")

    if endpoint and key:
        safety_client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
        logging.info("Content Safety Client initialized.")
    else:
        safety_client = None
        logging.warning("Safety credentials missing. Skipping safety checks.")

def remove_emojis(text):
    """
    Removes characters in the Unicode Supplementary Planes (Emojis) 
    but KEEPS Basic Multilingual Plane (Burmese, English, etc.)
    """
    text = str(text)
    return re.sub(r'[\U00010000-\U0010ffff]', '', text).strip()

def check_safety(text):
    """
    Returns (True, None) if safe.
    Returns (False, Reason) if unsafe.
    """
    if not safety_client:
        return True, None
    
    try:
        request = AnalyzeTextOptions(text=text)
        response = safety_client.analyze_text(request)
        
        # Check if any severity > 0 is detected (Self-Harm, Violence, etc.)
        for item in response.categories_analysis:
            if item.severity > 0:
                return False, f"{item.category} detected"
        return True, None
    except Exception as e:
        logging.error(f"Safety check failed: {e}")
        # Fail open (allow processing if API fails) so we don't break the app
        return True, None 

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        # 1. Get Inputs
        input_list = data.get("inputs", [])
        if isinstance(input_list, str): input_list = [input_list]
        
        display_list = data.get("original_text", input_list)
        
        results = []

        # 2. Process each post
        for trans_text, orig_text in zip(input_list, display_list):
            
            clean_display_text = remove_emojis(orig_text)

            # --- STEP A: CHECK AZURE SAFETY ---
            is_safe, reason = check_safety(trans_text)

            if not is_safe:
                # If Azure flags it as Self-Harm/Violence, we do NOT block it.
                # Instead, we force the prediction to '6' (Suicidal).
                # This ensures it shows up as CRITICAL (Level 4) on your graph.
                results.append({
                    "text": clean_display_text,
                    "prediction": 6, 
                    "note": f"Azure Flagged: {reason}"
                })
                continue

            # --- STEP B: RUN BERT MODEL (If Safe) ---
            inputs = tokenizer(
                [trans_text], 
                padding=True, 
                truncation=True, 
                max_length=200, 
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}

            with torch.no_grad():
                outputs = model(**inputs)
            
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            
            results.append({
                "text": clean_display_text,
                "prediction": pred_id
            })
        
        return results
        
    except Exception as e:
        return {"error": str(e)}