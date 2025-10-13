import pandas as pd

if __name__ == "__main__":    # Example usage
    # print(f"Gini Impurity: {impurity}")
    data = pd.read_csv('../../data/lab_2/playgolf.csv')
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    print(outlook)
    print(outlook.shape)
