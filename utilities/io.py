import pandas as pd


def save_pd_cvs(name, data):
    # Take dict of data to save to csv caled name
    df = pd.DataFrame(data=data)
    df.to_csv(name + ".csv", sep=',', index=False)
    return 0
