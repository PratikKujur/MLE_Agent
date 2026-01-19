from langchain_groq import ChatGroq
from typing_extensions import TypedDict
import os
from Tasks import ProblemType,EDAReport,EDAStrategy

class State(TypedDict):
    Domain_expert: dict
    Dataset_profiler: str
    EDA_Resonner: str
    EDA_Executer: dict
    EDA_report_generator: str

class EDA_Agents:
    def __init__(self):
         self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )

    def Domain_expert(self,state: State, df, columns) -> dict:
        result = self.llm.with_structured_output(ProblemType).invoke(
            f"""You are a Domain expert, you have to analyze the dataset and column names
    and identify the type of problem we are trying to solve ("regression", "classification", "clustering", "unknown"),
    what is the target variable if not a clustering problem, and provide the confidence scores for each problem type.

    dataset_sample:{df.to_dict()}
    dataset_columns:{columns}

    Respond with a JSON object with the following fields:
    - problem_type: one of "regression", "classification", "clustering"
    - target_variable: the name of the target variable (if applicable)
    - confidence_score_regression: confidence as a float between 0 and 1
    - confidence_score_classification: confidence as a float between 0 and 1
    - confidence_score_clustering: confidence as a float between 0 and 1
    """
        )
        return {
            "Domain_expert": result.model_dump()
            if hasattr(result, "model_dump")
            else result
        }


    def Dataset_profiling(self,state: State) -> dict:
        return {
            "EDA_report_generator": self.llm.invoke(f"""
    Your are Dataset Profiler, you have to generate a detailed dataset profiling report based on the following basic EDA analysis:
                        Basic_EDA_analysis:{state["Dataset_profiler"]}""").content
        }

    def EDA_Strategy_Generator(self,state: State) -> dict:
        result = self.llm.with_structured_output(EDAStrategy).invoke(f"""
        You are an EDA specialist, you have to genrate an EDA strategy based on the following domain expertise.
        Report: It should be a detailed plan outlining the EDA approach.
        focus_areas: List of key areas to focus on during EDA,should be amoung (descriptive_analysis, correlation_analysis, outlier_detection,feature_ranking).
        red_flags: List of potential issues or concerns to watch out for.
        analysis_to_run: List of specific analyses to be conducted.
        analysis_to_skip: List of analyses that are not necessary and takes lots of computation.
        priority_order: List of priorities for the analyses.
                                        
        Domain_expertise:{state["Domain_expert"]}""")
        return {
            "EDA_Resonner": result.model_dump() if hasattr(result, "model_dump") else result
        }
    def EDA_Report(self,df_sample,state:State)->dict:
        problem_type = state["Domain_expert"].get("problem_type", "unknown") if isinstance(state["Domain_expert"], dict) else "unknown"
        
        result = self.llm.with_structured_output(EDAReport).invoke(f"""You are an EDA Report writer specialist. Create a professional EDA report.

Problem Type: {problem_type}

Generate a JSON report with:
- Report: A concise summary of EDA findings
- key_insights: Main insights discovered
- risks: Potential risks identified  
- modeling_implications: Impact on modeling
- next_steps: Recommended next steps

Keep responses concise and focused.""")
        
        return {
            "EDA_Report": result.model_dump() if hasattr(result, "model_dump") else result
        }