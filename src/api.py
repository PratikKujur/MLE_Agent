from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from main import get_eda

app = FastAPI()

# Add CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class FileInput(BaseModel):
    file_path: Optional[str] = None

class EDA_Agent_Response(BaseModel):
    Domain_expert: Optional[dict] = None
    Dataset_profiler: Optional[str] = None
    EDA_Resonner: Optional[str] = None
    EDA_report_generator: Optional[str] = None

@app.post("/upload")
async def upload_file(file_input: FileInput):
    try:
        if not file_input.file_path:
            return {
                "error": "No file path provided",
                "Domain_expert": None,
                "Dataset_profiler": None,
                "EDA_Resonner": None,
                "EDA_report_generator": None
            }
        
        if not os.path.exists(file_input.file_path):
            return {
                "error": f"File not found: {file_input.file_path}",
                "Domain_expert": None,
                "Dataset_profiler": None,
                "EDA_Resonner": None,
                "EDA_report_generator": None
            }
        
        print(f"Starting EDA analysis for file: {file_input.file_path}")
        result = get_eda(file_path=file_input.file_path)
        
        if result is None:
            return {
                "error": "get_eda() returned None",
                "Domain_expert": None,
                "Dataset_profiler": None,
                "EDA_Resonner": None,
                "EDA_report_generator": None
            }
        
        print(f"EDA analysis completed. Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        # Ensure we return a proper response
        response_data = {
            "Domain_expert": result.get("Domain_expert"),
            "Dataset_profiler": result.get("Dataset_profiler"),
            "EDA_Resonner": result.get("EDA_Resonner"),
            "EDA_report_generator": result.get("EDA_report_generator"),
        }
        
        if "error" in result:
            response_data["error"] = result["error"]
        
        print(f"Returning response: {response_data}")
        return response_data
        
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Exception: {str(e)}",
            "Domain_expert": None,
            "Dataset_profiler": None,
            "EDA_Resonner": None,
            "EDA_report_generator": None
        }

@app.get("/health")
async def health_check():
    return {"status": "ok"}