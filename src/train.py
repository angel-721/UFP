#!/usr/bin/env python3
import argparse
import pickle

import pandas as pd
from sklearn import pipeline
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

data = "../data/ufc-master-train.csv"
test = "../data/ufc-master-test.csv"
test2 = "../data/main-events.csv"
label = "Winner"
features = ["R_odds", "B_odds", "R_wins",
            "B_wins", "lose_streak_dif", "win_streak_dif",
            "age_dif", "height_dif", "reach_dif"]


def parseArgs():
    parser = argparse.ArgumentParser(
        prog="UFC Winner predictor", description="A predictor for UFC fights")
    parser.add_argument('action', default="train", choices=[
                        "train", "predict", "score", "load"], nargs="?")
    parser.add_argument("--show-test", '-t', default=0, type=int)
    parser.add_argument("--show-test-2", '-t2', default=0, type=int)
    parser.add_argument(
        "--show-cash-made", '-c', default=0, type=int, help="Calculate earnings from bet amount user puts in")
    parser.add_argument("--model-name", '-m', default="model.pkl", type=str)
    args = parser.parse_args()

    return args


def loadData(fileName: str, label: str, features: list[str]):
    df = pd.read_csv(fileName)
    X = df[features]
    Y = df[label]

    return X, Y


def loadDataNoDrop(fileName: str):
    df = pd.read_csv(fileName)
    return df


xTrain, yTrain = loadData(data, label, features)
xTest, yTest = loadData(test, label, features)
xTest2, yTest2 = loadData(test2, label, features)
xTest2NoDrop = loadDataNoDrop(test2)
xTestNoDrop = loadDataNoDrop(test)
xNoDrop = loadDataNoDrop(data)


def makePipeline(cvFits):
    selector = SelectPercentile(mutual_info_classif, percentile=80)
    c1 = SVC(kernel="rbf")
    c2 = GradientBoostingClassifier(n_estimators=100)
    c3 = KNeighborsClassifier(3)
    estimatorsList = [("SVC", c1), ("GBC", c2), ("KNN", c3)]
    clf = VotingClassifier(estimators=estimatorsList)
    pipeline = Pipeline([
        ("Encode", OneHotEncoder(handle_unknown="ignore")),
        ("Impute", SimpleImputer()),
        ("FeatureSelection", selector),
        ("Classifier", clf)
    ])
    param_grid = {
        'FeatureSelection__percentile': [30, 40, 50, 70, 90, 100],
        'Classifier__GBC__n_estimators': [50, 100, 200],
        'Classifier__GBC__learning_rate': [0.01, 0.1, 0.5],
        'Classifier__SVC__C': [0.0025, 0.1, 1, 10],
        'Classifier__SVC__gamma': [0.1, 1, 'scale', 'auto'],
        'Classifier__KNN__n_neighbors': [3, 5, 7],
        'Classifier__KNN__weights': ['uniform', 'distance']
    }

    gridSearchPipeline = GridSearchCV(
        pipeline, param_grid=param_grid, cv=cvFits)

    return gridSearchPipeline, numFits, cvFits


def trainAndSave(pipeline, xTrain, yTrain, modelName, fits, cvFits):
    print("Fitting model over", fits, "fits. Over", cvFits, "folds")
    pipeline.fit(xTrain, yTrain)
    print(pipeline.best_score_)
    print(pipeline.best_params_)
    print(pipeline.feature_names_in_)
    pickle.dump(pipeline, open(modelName, 'wb'))
    print("Model saved to", modelName)


def score(modelName, xTrain, yTrain, xTest, yTest,
          xTest2, yTest2, showTest, showTest2, cashMade):
    model = pickle.load(open(modelName, 'rb'))
    trainScore = model.score(xTrain, yTrain)
    print("Loaded model:", modelName)
    print("Training score is", trainScore)
    if cashMade:
        calculateCashMade(modelName, xTest, xNoDrop, cashMade)
    if showTest != 0:
        testScore = model.score(xTest, yTest)
        print("Testing score for test 1 is", testScore)
        if cashMade:
            calculateCashMade(modelName, xTest, xTestNoDrop, cashMade)
    if showTest2 != 0:
        testScore2 = model.score(xTest2, yTest2)
        print("Testing score for test 2 is", testScore2)
        if cashMade:
            calculateCashMade(modelName, xTest2, xTest2NoDrop, cashMade)


def calculateCashMade(modelName, x, xNames, betAmount):
    model = pickle.load(open(modelName, 'rb'))
    winners = model.predict(x)
    cashMade = 0
    odds = 0
    for i in range(len(winners)):
        if winners[i] == "Blue":
            if (winners[i] == xNames.loc[i, "Winner"]):
                odds = int(xNames.loc[i, "B_odds"])
                if (odds < 0):
                    earnings = (100 / odds) * betAmount + betAmount
                else:
                    earnings = (odds / 100) * betAmount + betAmount
                cashMade += earnings
            else:
                cashMade -= betAmount
        else:
            if (winners[i] == xNames.loc[i, "Winner"]):
                odds = int(xNames.loc[i, "R_odds"])
                if (odds < 0):
                    earnings = (100 / odds) * betAmount + betAmount
                else:
                    earnings = (odds / 100) * betAmount + betAmount
                cashMade += earnings
            else:
                cashMade -= betAmount
    print("Total cash from bets is:", cashMade)


def predict(modelName, x, xNames, cashMade):
    model = pickle.load(open(modelName, 'rb'))
    winners = model.predict(x)
    for i in range(len(winners)):
        if winners[i] == "Blue":
            print(xNames.iloc[i][1])  # 3 is the index of the B_fighter column
        else:
            print(xNames.iloc[i][0])  # 0 is the index of the R_fighter column
    if cashMade:
        calculateCashMade(modelName, x, xNames, cashMade)


def showModel(modelName):
    model = pickle.load(open(modelName, 'rb'))
    print(model.best_score_)
    print(model.best_params_)
    print(model.feature_names_in_)
    print(model.best_estimator_)


def main(args):

    if args.action == "train":
        pipeline, fits, folds = makePipeline(15)
        trainAndSave(pipeline, xTrain, yTrain, args.model_name, fits, folds)
    if args.action == "score":
        score(args.model_name, xTrain, yTrain, xTest,
              yTest, xTest2, yTest2, args.show_test, args.show_test_2, args.show_cash_made)
    if args.action == "predict":
        predict(args.model_name, xTest2, xTest2NoDrop, args.show_cash_made)
    if args.action == "load":
        showModel(args.model_name)


if __name__ == "__main__":
    main(parseArgs())
