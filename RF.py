import gc

import dask.dataframe as dd
import dask_ml.model_selection as dask_ms
from dask.distributed import Client
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from db_utils import get_data, get_canadian_data

N_WORKERS = 8
THREAD_P_WORKER = 4


def main(canadian: bool = False) -> None:
    if canadian:
        df = get_canadian_data()
    else:
        df = get_data()
    
    le = LabelEncoder()
    X = df.drop(columns=["genre_tzanetakis"])
    y = pd.Series(le.fit_transform(df["genre_tzanetakis"]), index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # client = Client(n_workers=N_WORKERS, threads_per_worker=THREAD_P_WORKER)
    # del df, X, y
    # client.run(gc.collect)
    # X_train = dd.from_pandas(X_train, npartitions=N_WORKERS)
    # y_train = dd.from_pandas(y_train, npartitions=N_WORKERS)
    # rf_pipe = Pipeline([("model", RandomForestClassifier(n_jobs=1, class_weight="balanced"))])
    # scorer = make_scorer(f1_score, average="weighted")
    # param_dist = {
    #     "model__n_estimators": randint(50, 500),
    #     "model__max_depth": [10, 15, 30],
    #     "model__min_samples_split": randint(2, 20),
    #     "model__min_samples_leaf": randint(1, 10),  
    # }
    # random_search = dask_ms.RandomizedSearchCV(
    #     rf_pipe,
    #     param_distributions=param_dist,
    #     n_iter=20,
    #     cv=3,
    #     scoring=scorer
    # )
    # random_search.fit(X_train, y_train)
    # print(f"Best params: {random_search.best_params_}")
    # print(f"Best val accuracy: {random_search.best_score_*100:.2f}%")

    rf_model = RandomForestClassifier(
            n_jobs=-1,
            class_weight="balanced",
            max_depth=30,
            min_samples_leaf=2,
            min_samples_split=3,
            n_estimators=97,
        )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print(f"Total Test F1: {f1_score(y_test, y_pred, average='weighted')}")
    print(f"Test F1 Per Category: {f1_score(y_test, y_pred, average=None)}")

    features = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    features = features.sort_values(ascending=False)

    plt.figure(figsize=(12, 5))
    features.plot(kind="bar", color="steelblue")
    plt.title("Random Forest — Feature Importance")
    plt.xticks(rotation=45, ha="right")

    plt.savefig(f"./images/{'canadian_' if canadian else ''}feature_import.png", bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":
    main(True)
