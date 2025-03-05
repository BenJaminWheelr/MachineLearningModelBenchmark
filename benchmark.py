import math
import pandas as pd
import argparse
import sys
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def benchmark(dataFrame, classification, regression, xColumns, yColumn):
    dataFrame = nullCheck(dataFrame, xColumns, yColumn)
    y = dataFrame[yColumn].values.ravel()
    if xColumns:
        x = dataFrame[xColumns]
    else:
        x = dataFrame.drop(columns=yColumn)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)
    if classification:
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB


        modelList = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, KNeighborsClassifier, GaussianNB]
        scaler = StandardScaler()
        for model in modelList:
            currentModel = model()
            predictions, elapsedTime = trainModel(currentModel, xTrain, yTrain, xTest)
            print(f"\n{model.__name__} ACCURACY SCORE UNSCALED: {accuracy_score(yTest, predictions)} | TRAINING TIME: {elapsedTime}")
            xTrainScaled = scaler.fit_transform(xTrain)
            xTestScaled = scaler.transform(xTest)
            scaledPredictions, elapsedTime = trainModel(currentModel, xTrainScaled, yTrain, xTestScaled)
            print(f"{model.__name__} ACCURACY SCORE SCALED: {accuracy_score(yTest, scaledPredictions)} | TRAINING TIME: {elapsedTime}")
    if regression:
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        modelList = [LinearRegression, SVR, DecisionTreeRegressor, RandomForestRegressor]
        scaler = StandardScaler()
        for model in modelList:
            currentModel = model()
            predictions, elapsedTime = trainModel(currentModel, xTrain, yTrain, xTest)
            print(f"\n{model.__name__} RMSE SCORE UNSCALED: {math.sqrt(mean_squared_error(yTest, predictions))} | TRAINING TIME: {elapsedTime}")
            xTrainScaled = scaler.fit_transform(xTrain)
            xTestScaled = scaler.transform(xTest)
            scaledPredictions, elapsedTime = trainModel(currentModel, xTrainScaled, yTrain, xTestScaled)
            print(f"{model.__name__} RMSE SCORE SCALED: {math.sqrt(mean_squared_error(yTest, predictions))} | TRAINING TIME: {elapsedTime}")
        print()
        for i in range(2, 6):
            polynomialModel = make_pipeline(PolynomialFeatures(degree=i), LinearRegression())
            startTime = time.time()
            polynomialModel.fit(xTrain, yTrain)
            endTime = time.time()
            elapsedTime = endTime - startTime
            predictions = polynomialModel.predict(xTest)
            print(f"Polynomial Regression RMSE (DEGREE: {i}): {math.sqrt(mean_squared_error(yTest, predictions))} | TRAINING TIME: {elapsedTime}")
        alphaValues = [0.01, 0.1, 1, 10, 100]
        regularizationModels = [Ridge, Lasso]
        for value in alphaValues:
            for model in regularizationModels:
                currentModel = model(alpha=value)
                predictions, elapsedTime = trainModel(currentModel, xTrain, yTrain, xTest)
                print(f"\n{model.__name__} (ALPHA={value}) RMSE SCORE UNSCALED: {math.sqrt(mean_squared_error(yTest, predictions))} | TRAINING TIME: {elapsedTime}")
                xTrainScaled = scaler.fit_transform(xTrain)
                xTestScaled = scaler.transform(xTest)
                scaledPredictions, elapsedTime = trainModel(currentModel, xTrainScaled, yTrain, xTestScaled)
                print(f"{model.__name__}(ALPHA={value}) RMSE SCORE SCALED: {math.sqrt(mean_squared_error(yTest, predictions))} | TRAINING TIME: {elapsedTime}")


def trainModel(currentModel, xTrain, yTrain, xTest):
    startTime = time.time()
    currentModel.fit(xTrain, yTrain)
    endTime = time.time()
    predictions = currentModel.predict(xTest)
    elapsedTime = endTime - startTime
    return predictions, elapsedTime


def nullCheck(dataFrame, xColumns, yColumns):
    print("\nCOLUMN NAME: NULL COUNT")
    nullCountTotal = 0
    columnsWithNull = []
    if xColumns:
        for column in xColumns + yColumns:
            nullCount = dataFrame[column].isna().sum()
            nullCountTotal += nullCount
            if nullCount > 0:
                columnsWithNull.append(column)
            print(f"\t{column}: {nullCount}")
    else:
        for column in dataFrame.columns:
            nullCount = dataFrame[column].isna().sum()
            nullCountTotal += nullCount
            if nullCount > 0:
                columnsWithNull.append(column)
            print(f"\t{column}: {nullCount}")
    if nullCountTotal > 0:
        redText("There are missing values in the dataset.")
        cleaningMethod = input("How would you like to replace these values?\nReplace with Mean (mean)\nReplace with Median (median)\nDrop rows with Null (drop)\nType Anything else to quit: ")
        if cleaningMethod == "mean":
            cleanNull(dataFrame, columnsWithNull, "mean")
        elif cleaningMethod == "median":
            cleanNull(dataFrame, columnsWithNull, "median")
        elif cleaningMethod == "drop":
            dataFrame = dataFrame.dropna()
        else:
            redText(f"Not a valid option. Quitting.")
            sys.exit(1)
    return dataFrame


def cleanNull(dataFrame, columnList, method):
    for column in columnList:
        print(f"FILLING {column} WITH {method.upper()}")
        methodValue = getattr(dataFrame[column], method)()
        dataFrame[column] = dataFrame[column].fillna(methodValue)




def redText(message):
    print(f"\033[31m{message}\033[0m")


def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="A script to benchmark the performance of different machine learning models.")
    parser.add_argument('file', type=str, help="Path to the dataset CSV file.")
    parser.add_argument('--classification', action="store_true", help="Specifies a classification machine learning model.")
    parser.add_argument('--regression', action="store_true", help="Specifies a regression machine learning model.")
    parser.add_argument('--xColumns', type=str, nargs="+", metavar='COLUMN/S', help="Specifies the features to use to predict.")
    parser.add_argument('--yColumn', type=str, nargs="+", metavar='COLUMN', required=True, help="Specifies the label/value to predict.")

    args = parser.parse_args()
    print("PARSING DATA...")
    lineCount = sum(1 for _ in open(args.file))
    df = pd.read_csv(args.file, iterator=True, chunksize=10000)
    df = pd.concat(tqdm(df, total=lineCount // 10000))

    benchmark(df, args.classification, args.regression, args.xColumns, args.yColumn)


if __name__ == "__main__":
    main()
