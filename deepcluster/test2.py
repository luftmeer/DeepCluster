import pandas as pd


def main():
    df = pd.read_csv('training_metrics.csv')
    df.head()


if __name__ == '__main__':
    main()
