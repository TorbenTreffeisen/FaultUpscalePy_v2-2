
predefined_paths_db = {'db': ['input_files_db']}

predefined_paths_comp = {'comp': ['input_files_comp']}

source_keys = {'db': ['predefined_paths_db'],
               'comp': ['predefined_paths_comp']}


def list_keys(dic):
    li = []
    for key in dic:
        li.append(key)
    return li


def predef_paths(source, model):
    if source == 'db':
        path = predefined_paths_db[source][0] + '/' + model + '.csv'

    elif source == "comp":
        path = predefined_paths_comp[source][0] + '/' + model + '.csv'

    else:
        pass

    return path


mod1key = list_keys(predefined_paths_db)
mod2key = list_keys(predefined_paths_comp)


