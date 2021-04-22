import os
import re


from dic_def import *
from plotting import *
from scipy.interpolate import griddata
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def files_in_folder(folderpath):
    outwrite = []

    for root, dirs, files in os.walk(folderpath):
        for filename in files:
            nw_name = re.sub('.csv', '', filename)
            outwrite.append(nw_name)

    return outwrite

def files_in_folder_png(folderpath):
    outwrite = []

    for root, dirs, files in os.walk(folderpath):
        for filename in files:
            nw_name = re.sub('.png', '', filename)
            outwrite.append(nw_name)

    return outwrite


def pic_creation(var, models, folder, data_format, inp_out, key, min_val, max_val, cshades, cmap, xi, yi):
    for i in models:
        model_paths = predef_paths(var, i)

        # Calculations
        results_mod = ans_csvdataimp_complete(model_paths)

        # generate df for plotting
        xval, yval, zval, chead, output_title = plot_df(results_mod, inp_out, key)

        # define graduation
        user_graduation = np.linspace(min_val, max_val, cshades, endpoint=True)

        # define Colormap
        colm = colormap(cmap)

        # Plotting - general settings
        fig = plt.figure(figsize=(10, 10))
        spec = gridspec.GridSpec(ncols=1, nrows=1)

        # figure 1

        # model1
        fig_img = fig.add_subplot(spec[0, 0])
        # Contour plot
        zi = griddata((xval, yval), zval, (xi[None, :], yi[:, None]), method='linear')
        cf_f1_1 = plt.contourf(xi, yi, zi, user_graduation, cmap=colm)

        fig_img.axes.get_xaxis().set_visible(False)
        fig_img.axes.get_yaxis().set_visible(False)  # Achsen unsichtbar machen

        # plt.ylabel(i)

        # plt.title(output_title)

        # fig.colorbar(cf_f1_1)

        plt.savefig(folder + '_' + 'pic' + '/' + i + data_format, bbox_inches='tight')
        plt.close()

        # plt.show()