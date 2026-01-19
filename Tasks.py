from typing_extensions import TypedDict
from typing import Optional,Literal,List,Dict
from pydantic import BaseModel
import numpy as np
from sklearn.feature_selection import mutual_info_classif,mutual_info_regression

class State(TypedDict):
    Domain_expert: dict
    Dataset_profiler: str
    EDA_Resonner: str
    EDA_Executer: dict
    EDA_report_generator: str

class BasicEDA(BaseModel):
    shape: Optional[tuple] | None = None
    missing_values: Optional[dict] | None = None
    dtypes: Optional[dict] | None = None
    class_imbalance: Optional[dict] | None = None
    categorical_cardinality: Optional[Dict[str, int]] | None = None
    duplicate_rows: Optional[List[str]] | None = None
    constant_columns: Optional[List[str]] | None = None
    all_columns: Optional[List[str]] | None = None
    numeric_columns: Optional[List[str]] | None = None
    categorical_columns: Optional[List[str]] | None = None


class EDAStrategy(BaseModel):
    Report: str
    focus_areas: List[str]
    red_flags: List[str]
    analysis_to_run: List[str]
    analysis_to_skip: List[str]
    priority_order: List[str]


class ProblemType(BaseModel):
    problem_type: Literal["regression", "classification", "clustering"] = (
        "unknown"
    )
    target_variable: Optional[str] = None
    confidence_score_regression: Optional[float] = None
    confidence_score_classification: Optional[float] = None
    confidence_score_clustering: Optional[float] = None

class EDAReport(BaseModel):
    Report: str
    key_insights: List[str]
    risks: List[str]
    modeling_implications: List[str]
    next_steps: List[str]


class EDA_Tasks:
    def Dataset_profiling_regression(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        shape = df.shape
        missing_values = {k: int(v) for k, v in df.isnull().sum().items()}
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        class_imbalance = None
        cardinality = None
        duplicate_row = []
        constant_columns = [x for x in df.columns if len(np.unique(df[x])) == 1]
        all_columns = df.columns.tolist()
        numeric_cols = [
            x
            for x in df.columns
            if df[x].dtype in ["int64", "float64"] and x != target_variable
        ]
        categorical_cols = [
            x for x in df.columns if x not in numeric_cols and x != target_variable
        ]

        result = BasicEDA(
            shape=shape,
            missing_values=missing_values,
            dtypes=dtypes,
            class_imbalance=class_imbalance,
            categorical_cardinality=cardinality,
            duplicate_rows=duplicate_row,
            constant_columns=constant_columns,
            all_columns=all_columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

        return {"Dataset_profiler": result.json()}


    def Dataset_profiling_classification(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        shape = df.shape
        missing_values = {k: int(v) for k, v in df.isnull().sum().items()}
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        class_imbalance = {k: int(v) for k, v in df[target_variable].value_counts().items()}
        duplicate_row = []
        constant_columns = [x for x in df.columns if len(np.unique(df[x])) == 1]
        all_columns = df.columns.tolist()
        numeric_cols = [
            x
            for x in df.columns
            if df[x].dtype in ["int64", "float64"] and x != target_variable
        ]
        categorical_cols = [
            x for x in df.columns if x not in numeric_cols and x != target_variable
        ]
        cardinality = {x: int(df[x].nunique()) for x in categorical_cols}

        result = BasicEDA(
            shape=shape,
            missing_values=missing_values,
            dtypes=dtypes,
            class_imbalance=class_imbalance,
            categorical_cardinality=cardinality,
            duplicate_rows=duplicate_row,
            constant_columns=constant_columns,
            all_columns=all_columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

        return {"Dataset_profiler": result.json()}


    def Dataset_profiling_clustering(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        shape = df.shape
        missing_values = {k: int(v) for k, v in df.isnull().sum().items()}
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        class_imbalance = None
        duplicate_row = []
        constant_columns = [x for x in df.columns if len(np.unique(df[x])) == 1]
        all_columns = df.columns.tolist()
        numeric_cols = [
            x
            for x in df.columns
            if df[x].dtype in ["int64", "float64"] and x != target_variable
        ]
        categorical_cols = [
            x for x in df.columns if x not in numeric_cols and x != target_variable
        ]
        cardinality = {x: int(df[x].nunique()) for x in categorical_cols}

        result = BasicEDA(
            shape=shape,
            missing_values=missing_values,
            dtypes=dtypes,
            class_imbalance=class_imbalance,
            categorical_cardinality=cardinality,
            duplicate_rows=duplicate_row,
            constant_columns=constant_columns,
            all_columns=all_columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
        )

        return {"Dataset_profiler": result.json()}


    def EDA_executer_descriptive(self,df,state:State) -> dict:
        return df.describe().to_dict()


    def EDA_executer_correlation(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        X = df.drop(columns=[target_variable])
        return X.corr().to_dict()


    def EDA_executer_outlier_detection(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        numeric_cols = state["Dataset_profiler"]["numeric_columns"]
        X = df.drop(columns=[target_variable])
        outlier_dict = {}

        for col in numeric_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
            outlier_dict[col] = outliers.to_dict()

        return outlier_dict


    def EDA_executer_feature_ranking(self,df, state: State) -> dict:
        target_variable = state["Domain_expert"]["target_variable"]
        problem_type = state["Domain_expert"]["problem_type"]
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        feature_ranking = {}

        if problem_type == "regression":
            mi_scores = mutual_info_regression(X, y)
        elif problem_type == "classification":
            mi_scores = mutual_info_classif(X, y)
        else:
            return {
                "feature_ranking": "Feature ranking not applicable for clustering or unknown problem types."
            }

        for col, score in zip(X.columns, mi_scores):
            feature_ranking[col] = float(score)

        sorted_ranking = dict(
            sorted(feature_ranking.items(), key=lambda item: item[1], reverse=True)
        )
        return {"feature_ranking": sorted_ranking}

   