#!/usr/bin/env python3
import argparse
import pickle

import pandas as pd
from sklearn import pipeline

featureList = ["R_odds", "B_odds", "R_wins",
               "B_wins", "lose_streak_dif", "win_streak_dif",
               "age_dif", "height_dif", "reach_dif"]


def parseArgs():
    parser = argparse.ArgumentParser(
        prog="UFC Winner predictor", description="A predictor for UFC fights")
    parser.add_argument('action', default="predict", choices=[
        "predict", "load"], nargs="?")
    parser.add_argument("--model-name", '-m',
                        default="./models/model1.pkl", type=str)
    parser.add_argument("--data-file", '-d',
                        default="./data/example.csv", type=str)
    args = parser.parse_args()
    return args


def loadData(fileName: str, label: str = "Winner", features: list[str] = featureList):
    df = pd.read_csv(fileName)
    X = df[features]
    Y = df[label]
    return X, Y


def loadDataNoDrop(fileName: str):
    df = pd.read_csv(fileName)
    return df


def predict(modelName, dataFile):
    x, _ = loadData(dataFile)
    xNames = loadDataNoDrop(dataFile)
    model = pickle.load(open(modelName, 'rb'))
    winners = model.predict(x)
    for i in range(len(winners)):
        winner = "Null"
        if winners[i] == "Blue":
            # 3 is the index of the B_fighter column
            winner = xNames.iloc[i][1]
        else:
            # 0 is the index of the R_fighter column
            winner = xNames.iloc[i][0]
        print("\n", xNames.iloc[i][1], "Vs",
              xNames.iloc[i][0], "Winner:", winner)


def showModel(modelName):
    model = pickle.load(open(modelName, 'rb'))
    print(model.best_score_)
    print(model.best_params_)
    print(model.feature_names_in_)
    print(model.best_estimator_)


def main(args):

    if args.action == "predict":
        predict(args.model_name, args.data_file)
    if args.action == "load":
        showModel(args.model_name)


if __name__ == "__main__":
    main(parseArgs())
