
from dataimp import *
from filepaths import *


def plot_df(results, dic, key):
    df_res = ans_calc(results, dic[key][0])
    xval = df_res['x']
    yval = df_res['y']
    zval = df_res[dic[key][0]]
    chead = dic[key][0]
    output = dic[key][2]
    return xval, yval, zval, chead, output


def graduation(resultmod1, resultmod2, dic, key):
    df_mod1 = ans_calc(resultmod1, dic[key][0])
    df_mod2 = ans_calc(resultmod2, dic[key][0])
    max_mod1 = df_mod1[dic[key][0]].max()
    max_mod2 = df_mod2[dic[key][0]].max()
    if max_mod1 >= max_mod2:
        maxm = max_mod1
    else:
        maxm = max_mod2

    min_mod1 = df_mod1[dic[key][0]].min()
    min_mod2 = df_mod2[dic[key][0]].min()
    if min_mod1 <= min_mod2:
        minm = min_mod1
    else:
        minm = min_mod2

    return minm, maxm

def grad_db(db_var, model_list, dic, key):
    max_list = []
    min_list = []

    for i in model_list:
        model_paths = predef_paths(db_var, i)

        # Calculations
        results_mod = ans_csvdataimp_complete(model_paths)

        df_i = ans_calc(results_mod, dic[key][0])
        max_i = df_i[dic[key][0]].max()
        min_i = df_i[dic[key][0]].min()

        max_list.append(max_i)
        min_list.append(min_i)

    max_all = max(max_list)
    min_all = min(min_list)

    return min_all, max_all


def grad_all(min_db, max_db, min_comp, max_comp):

    if max_db >= max_comp:
        maxm = max_db
    else:
        maxm = max_comp

    if min_db <= min_comp:
        minm = min_db
    else:
        minm = min_comp

    return minm, maxm

def colormap(col):
    if col == 'RdYlBu':
        col = col + '_r'
    elif col == 'RdYlGn':
        col = col + '_r'
    elif col == 'RdBu':
        col = col + '_r'
    elif col == 'spectral':
        col = col + '_r'
    else:
        col = col
    return col



