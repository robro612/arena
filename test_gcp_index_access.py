from pathlib import Path
from dotenv import load_dotenv
import yaml
import os
import json
from models import ModelManager, MODEL_TO_CUDA_DEVICE

load_dotenv()
MODEL_TO_CUDA_DEVICE["Alibaba-NLP/gte-Qwen2-7B-instruct"] = "0"

def get_credentials():
    import tempfile
    creds_json_str = os.getenv("GCP_CREDENTIALS") # get json credentials stored as a string
    if creds_json_str is None:
        raise ValueError("GCP_CREDENTIALS not found in environment")

    # create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        temp.write(creds_json_str) # write in json format
        temp_filename = temp.name 
    
    return temp_filename

def load_model_meta_yaml(file_path: str | Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Hello, World!")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = get_credentials()
    model_meta = load_model_meta_yaml("/exp/rjha/arena/model_meta.yml")
    model_manager = ModelManager(model_meta, use_gcp_index=True, load_all=False)
    model = model_manager.load_model("Alibaba-NLP/gte-Qwen2-7B-instruct", device="cuda")
    q_embs = model.encode_queries(["what is the capital of France?"])

    
    

