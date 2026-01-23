# EDA Agent System

An intelligent Exploratory Data Analysis (EDA) system powered by LangChain and LangGraph that automatically analyzes datasets, identifies problem types, and generates comprehensive EDA reports.

## Features

- **Automatic Problem Type Detection**: Identifies whether your dataset is suited for regression, classification, or clustering
- **Intelligent EDA Strategy**: Generates customized analysis plans based on domain expertise
- **Comprehensive Profiling**: Provides detailed dataset statistics and insights
- **Multi-Agent Architecture**: Uses specialized agents for different aspects of analysis
- **Web Interface**: User-friendly Streamlit interface for easy interaction

## Architecture

The system uses a multi-agent workflow:
1. **Domain Expert**: Analyzes dataset to identify problem type and target variable
2. **Dataset Profiler**: Generates detailed profiling based on problem type
3. **EDA Strategy Generator**: Creates focused analysis plan
4. **EDA Executors**: Runs specific analyses (descriptive, correlation, outliers, feature ranking)
5. **Report Generator**: Compiles findings into actionable insights

## Prerequisites

- Python 3.8+
- GROQ API Key

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd eda-agent-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Required Dependencies

Create a `requirements.txt` file with:
```
fastapi
uvicorn
streamlit
requests
pandas
scikit-learn
langchain
langchain-groq
langgraph
python-dotenv
typing-extensions
pydantic
```

## Project Structure

```
.
├── main.py                 # Main workflow orchestration
├── Agents.py              # Agent definitions
├── Tasks.py               # Task definitions (not shown but required)
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit frontend
├── .env                   # Environment variables
├── uploads/               # Upload directory (auto-created)
└── README.md
```

## Usage

### Step 1: Start the FastAPI Backend

In one terminal, run:
```bash
uvicorn api:app --reload
```

The API will start at `http://127.0.0.1:8000`

You can verify it's running by visiting:
```
http://127.0.0.1:8000/health
```

### Step 2: Launch the Streamlit Interface

In another terminal, run:
```bash
streamlit run app.py
```

The web interface will open in your browser (usually at `http://localhost:8501`)

### Step 3: Upload and Analyze

1. Upload a CSV file through the Streamlit interface
2. Click "Upload dataset"
3. Wait for the analysis to complete (may take a few minutes)
4. Review the results:
   - **Domain Expert Analysis**: Problem type identification and confidence scores
   - **Dataset Profile**: Statistical summary and data quality metrics
   - **EDA Strategy**: Recommended analysis approach
   - **EDA Report**: Comprehensive findings and next steps

## API Endpoints

### POST `/upload`
Analyzes a dataset and returns EDA results.

**Request Body:**
```json
{
  "file_path": "path/to/your/dataset.csv"
}
```

**Response:**
```json
{
  "Domain_expert": {
    "problem_type": "regression",
    "target_variable": "target",
    "confidence_score_regression": 0.95,
    "confidence_score_classification": 0.05,
    "confidence_score_clustering": 0.0
  },
  "Dataset_profiler": "Detailed profiling text...",
  "EDA_Resonner": {
    "report": "Analysis strategy...",
    "focus_areas": ["descriptive_analysis", "correlation_analysis"],
    "red_flags": ["Missing values detected"],
    "analysis_to_run": ["correlation_matrix", "distribution_plots"],
    "analysis_to_skip": ["complex_transformations"],
    "priority_order": ["descriptive_analysis", "correlation_analysis"]
  },
  "EDA_report_generator": "Final comprehensive report..."
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Configuration

### Timeout Settings
The default analysis timeout is 5 minutes (300 seconds). You can adjust this in `app.py`:
```python
response = requests.post(
    f"{BASE_URL}/upload",
    json={"file_path": file_path},
    timeout=300  # Adjust timeout here
)
```

### Model Configuration
The system uses `llama-3.1-8b-instant` from GROQ. You can change the model in `Agents.py`:
```python
self.llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"  # Change model here
)
```

## Troubleshooting

### "Cannot connect to API" Error
- Ensure the FastAPI server is running: `uvicorn api:app --reload`
- Check that the server is running on `http://127.0.0.1:8000`

### "Request timed out" Error
- Increase the timeout value in `app.py`
- For very large datasets, consider processing in batches

### Missing GROQ API Key
- Verify your `.env` file contains `GROQ_API_KEY=your_key_here`
- Ensure the `.env` file is in the project root directory

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify you have the `Tasks.py` file (referenced but not provided)

## Notes

- The system creates an `uploads/` directory automatically for temporary file storage
- CSV files are the only supported format currently
- Analysis results are stored in session state and cleared on page refresh
- The system uses LangGraph for workflow orchestration and LangChain for LLM interactions

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Add appropriate error handling
- Update documentation for new features

## License

[Add your license here]

## Support

For issues or questions, please [open an issue](your-repo-url/issues) on GitHub.
