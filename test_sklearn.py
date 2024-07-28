import sklearn_utils as sk
import pandas as pd

def main():
    data = pd.read_csv( "fake reviews dataset.csv" )
    sk.learn( data.iloc )
    return

if __name__ == "__main__":
    main()