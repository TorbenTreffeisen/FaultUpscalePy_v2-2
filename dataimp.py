import pandas as pd


def ans_csvdataimp_complete(name):      # list_files, result_parameter, list_keys
    df = pd.read_csv(name)          # Einlesen CSV
    df = df.apply(pd.to_numeric)                               # Datatype to numeric ändern.

    return df


def ans_calc(results, parameter):
    df_calc = pd.concat([results['x'], results['y'], results[parameter]], axis=1)
    df_calc.columns = ['x', 'y', parameter]

    return df_calc
