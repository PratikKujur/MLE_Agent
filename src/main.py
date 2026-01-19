import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from Tasks import EDA_Tasks, State
from Agents import EDA_Agents
from sklearn.datasets import load_diabetes
from langchain_groq import ChatGroq
import pandas as pd

load_dotenv()
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
# os.environ["LANGSMITH_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
# os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")

eda_agents = EDA_Agents()
eda_tasks = EDA_Tasks()

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")


llm_profiled_tools = llm.bind_tools(
    [
        eda_tasks.Dataset_profiling_regression,
        eda_tasks.Dataset_profiling_classification,
        eda_tasks.Dataset_profiling_clustering,
    ]
)
llm_eda_tools = llm.bind_tools(
    [
        eda_tasks.EDA_executer_descriptive,
        eda_tasks.EDA_executer_correlation,
        eda_tasks.EDA_executer_outlier_detection,
        eda_tasks.EDA_executer_feature_ranking,
    ]
)


def basic_tranformation(df):
    return df.head(5), df.columns.tolist()


def get_eda(file_path:str):
    df=pd.read_csv(file_path)
    df_sample, columns = basic_tranformation(df)

    workflow = StateGraph(State)

    workflow.add_node(
        "Problem_type", lambda state: eda_agents.Domain_expert(state, df_sample, columns)
    )
    workflow.add_node(
        "Dataset_profiling_regression",
        lambda state: eda_tasks.Dataset_profiling_regression(df=df, state=state),
    )
    workflow.add_node(
        "Dataset_profiling_classification",
        lambda state: eda_tasks.Dataset_profiling_classification(df=df, state=state),
    )
    workflow.add_node(
        "Dataset_profiling_clustering",
        lambda state: eda_tasks.Dataset_profiling_clustering(df=df, state=state),
    )

    workflow.add_node("Dataset_profiling_report", eda_agents.Dataset_profiling)
    workflow.add_node("EDA_Strategy_Generator", eda_agents.EDA_Strategy_Generator)
    workflow.add_node(
        "EDA_executer_correlation",
        lambda state: eda_tasks.EDA_executer_correlation(df=df, state=state),
    )
    workflow.add_node(
        "EDA_executer_outlier_detection",
        lambda state: eda_tasks.EDA_executer_outlier_detection(df=df, state=state),
    )
    workflow.add_node(
        "EDA_executer_feature_ranking",
        lambda state: eda_tasks.EDA_executer_feature_ranking(df=df, state=state),
    )
    workflow.add_node(
        "EDA_executer_descriptive",
        lambda state: eda_tasks.EDA_executer_descriptive(df=df, state=state),
    )

    workflow.add_node("EDA_Report", lambda state: eda_agents.EDA_Report(df_sample, state))

    workflow.add_edge(START, "Problem_type")
    workflow.add_conditional_edges(
        "Problem_type",
        lambda state: state["Domain_expert"].get("problem_type", "unknown") if isinstance(state["Domain_expert"], dict) else "unknown",
        {
            "regression": "Dataset_profiling_regression",
            "classification": "Dataset_profiling_classification",
            "clustering": "Dataset_profiling_clustering",
        },
    )

    workflow.add_edge("Dataset_profiling_regression", "Dataset_profiling_report")
    workflow.add_edge("Dataset_profiling_classification", "Dataset_profiling_report")
    workflow.add_edge("Dataset_profiling_clustering", "Dataset_profiling_report")
    workflow.add_edge("Dataset_profiling_report", "EDA_Strategy_Generator")

    workflow.add_conditional_edges(
        "EDA_Strategy_Generator",
        lambda state: state["EDA_Resonner"].get("focus_areas", ["descriptive_analysis"])[0] if isinstance(state["EDA_Resonner"], dict) else "descriptive_analysis",
        {
            "descriptive_analysis": "EDA_executer_descriptive",
            "correlation_analysis": "EDA_executer_correlation",
            "outlier_detection": "EDA_executer_outlier_detection",
            "feature_ranking": "EDA_executer_feature_ranking",
        },
    )

    workflow.add_edge("EDA_executer_descriptive", "EDA_Report")
    workflow.add_edge("EDA_executer_correlation", "EDA_Report")
    workflow.add_edge("EDA_executer_outlier_detection", "EDA_Report")
    workflow.add_edge("EDA_executer_feature_ranking", "EDA_Report")

    workflow.add_edge("EDA_Report", END)

    chain = workflow.compile()

    initial_state: State = {
        "Domain_expert": {},
        "Dataset_profiler": "",
        "EDA_Resonner": "",
        "EDA_Executer": {},
        "EDA_report_generator": "",
    }

    try:
        result = chain.invoke(initial_state)
        return result
    except Exception as e:
        return {
            "error": str(e),
            "Domain_expert": {},
            "Dataset_profiler": "",
            "EDA_Resonner": "",
            "EDA_Executer": {},
            "EDA_report_generator": "",
        }
# print("Domain expert:", result["Domain_expert"], "\n")
# print("Domain profiler:", result["Dataset_profiler"], "\n")
# print("EDA strategy:", result["EDA_Resonner"], "\n")
# print("EDA_Report:", result["EDA_report_generator"])
