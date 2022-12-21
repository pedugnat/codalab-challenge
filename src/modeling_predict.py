"""
Main file for running the prediction task.
"""
# pylint: disable=missing-function-docstring
#!/usr/bin/env python
# coding: utf-8

# !pip install jours-feries-france -q
# !pip install vacances-scolaires-france -q

import os
import shutil
import warnings

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates

warnings.filterwarnings("ignore")

TARGETS = ["Available", "Charging", "Passive", "Other"]


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_feat_importance(clf, features):
    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importances_, features.columns)), columns=["Value", "Feature"]
    )

    plt.figure(figsize=(12, 4))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False),
    )
    plt.tight_layout()
    plt.show()


# data loading & preprocessing
df_data = pd.read_csv("../data/public_data/train.csv")
df_test = pd.read_csv("../data/public_data/test.csv")

df_data["date"] = pd.to_datetime(df_data["date"])
df_test["date"] = pd.to_datetime(df_test["date"])

df_train = df_data[(df_data["date"] > "2020-05-30")]

df_train["day"] = pd.to_datetime(df_train["date"].dt.date)
df_test["day"] = pd.to_datetime(df_test["date"].dt.date)


def sae(y_true, y_pred):
    """Sum of Absolute errors"""
    return sum(abs(y_true - y_pred))


# # Modeling

# ## Features
jf = list(JoursFeries.for_year(2020).values()) + list(
    JoursFeries.for_year(2021).values()
)

holidays = SchoolHolidayDates()
hd = [
    k for k, v in holidays.holidays_for_year(2020).items() if v["vacances_zone_c"]
] + [k for k, v in holidays.holidays_for_year(2021).items() if v["vacances_zone_c"]]

df_train["is_jf"] = df_train["date"].dt.date.isin(jf)
df_train["is_hd"] = df_train["date"].dt.date.isin(hd)

df_test["is_jf"] = df_test["date"].dt.date.isin(jf)
df_test["is_hd"] = df_test["date"].dt.date.isin(hd)


# use monthly historical temperatures
# (not very informative but helps capturing seasonality effect)

df = pd.read_csv("../data/paris-historical-temperature.csv", sep=";")
df["day"] = pd.to_datetime(df["observ_date"], dayfirst=True) + pd.tseries.offsets.Day(1)
df = df[df["day"] > "2020-01-01"]
df = df.set_index("day")
df = (
    df.reindex(pd.date_range("2020", "2022"))
    .ffill(limit=31)
    .reset_index()
    .rename(columns={"index": "day"})
)


# add temperatures and one hot encode area
df_train = pd.merge(df_train, df[["day", "avg_day", "avg_night"]], on=["day"]).drop(
    "Station", axis=1
)

df_train = pd.concat(
    (df_train.drop("area", axis=1), pd.get_dummies(df_train["area"])), axis=1
)

df_test = pd.merge(df_test, df[["day", "avg_day", "avg_night"]], on=["day"]).drop(
    "Station", axis=1
)

df_test = pd.concat(
    (df_test.drop("area", axis=1), pd.get_dummies(df_test["area"])), axis=1
)

# training loop

y_preds, models = [], []

train_feats = df_train.drop(TARGETS + ["date", "day", "trend"], axis=1)
test_feats = df_test.drop(["date", "day", "trend"], axis=1)

for target in tqdm(TARGETS):

    train_target = df_train[target]

    lgbm = lightgbm.LGBMRegressor(verbose=0)
    lgbm.fit(
        train_feats,
        train_target,
        eval_set=[
            (train_feats, train_target),
        ],
        eval_metric="l1",
        verbose=50,
        categorical_feature=["Postcode"],
    )

    y_preds.append(lgbm.predict(test_feats))
    models.append(lgbm)

    # plot_feat_importance(lgbm, train_feats)

preds = np.vstack(y_preds).T

test_station = pd.read_csv("../data/public_data/test.csv")

station_prediction = pd.concat(
    (test_station, pd.DataFrame(preds, columns=TARGETS)), axis=1
)[["date", "area", "Station"] + TARGETS]


# normalize in a simple but efficient way
for target in TARGETS:
    station_prediction[target] = (
        3 * station_prediction[target] / station_prediction.sum(axis=1)
    )


mkdir("sample_result_submission")

station_prediction[["date", "area", "Station"] + TARGETS].round().to_csv(
    "../output/sample_result_submission/station.csv", index=False
)


# # Area Level

# aggregate station prediction at the area level
area_prediction = (
    station_prediction.groupby(["date", "area"])[TARGETS].sum().reset_index()
)
area_prediction.to_csv("../output/sample_result_submission/area.csv", index=False)

# # Global Level

# aggregate station prediction at the area level
global_prediction = station_prediction.groupby(["date"])[TARGETS].sum().reset_index()
global_prediction.to_csv("../output/sample_result_submission/global.csv", index=False)

shutil.make_archive(
    "../output/sample_result_submission", "zip", "sample_result_submission"
)
