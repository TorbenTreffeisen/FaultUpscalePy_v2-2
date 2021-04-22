from qtpy import QtWidgets
from gui.mainwindow import Ui_MainWindow

from new_function import *


from dic_def import *
from plotting import *

import numpy as np

from scipy.interpolate import griddata
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.metrics import structural_similarity
import imutils
import cv2

app = QtWidgets.QApplication(sys.argv)

# Global Variables

db_folder = 'input_files_db'
comp_folder = 'input_files_comp'

db_var = 'db'
comp_var = 'comp'

fdb_1 = 'db1_pic/'
fdb_2 = 'db2_pic/'
fdb_3 = 'db3_pic/'

fcomp_1 = 'c1_pic/'
fcomp_2 = 'c2_pic/'
fcomp_3 = 'c3_pic/'

data_format = '.png'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pb_start.clicked.connect(self.initiate_comparison_plot)

        self.ui.pb_plot.clicked.connect(self.plot_results)
        self.ui.pb_comp.clicked.connect(self.comparison_img)

        self.ui.cbx_cat_c1.currentTextChanged.connect(self.cat_c1_changed)
        self.ui.cbx_cat_c2.currentTextChanged.connect(self.cat_c2_changed)
        self.ui.cbx_cat_c3.currentTextChanged.connect(self.cat_c3_changed)

    def initiate_comparison_plot(self):
        self.ui.cbx_comp.clear()

        model_comp = files_in_folder(comp_folder)
        inp_models = files_in_folder(db_folder)

        self.ui.cbx_comp.addItems(model_comp)

        # hier das abrufen und populieren der CBX

    def cat_c1_changed(self):
        self.ui.cbx_scat_c1.clear()

        cat1 = self.ui.cbx_cat_c1.currentText()
        res_pres = ['-']
        res_u = ['x', 'y', 'z']
        res_str = ['tot', 's1', 's2', 's3', 'sxx', 'syy', 'szz', 'sxy', 'syz', 'sxz']

        if cat1 == "pressure":
            self.ui.cbx_scat_c1.addItems(res_pres)
        elif cat1 == "u":
            self.ui.cbx_scat_c1.addItems(res_u)
        elif cat1 == "stress":
            self.ui.cbx_scat_c1.addItems(res_str)
        elif cat1 == "estress":
            self.ui.cbx_scat_c1.addItems(res_str)
        elif cat1 == "el. strain":
            self.ui.cbx_scat_c1.addItems(res_str)
        elif cat1 == "pl. strain":
            self.ui.cbx_scat_c1.addItems(res_str)
        elif cat1 == "tot. strain":
            self.ui.cbx_scat_c1.addItems(res_str)
        else:
            self.ui.cbx_scat_c1.addItems(" ")

    def cat_c2_changed(self):
        self.ui.cbx_scat_c2.clear()

        cat1 = self.ui.cbx_cat_c2.currentText()
        res_pres = ['-']
        res_u = ['x', 'y', 'z']
        res_str = ['tot', 's1', 's2', 's3', 'sxx', 'syy', 'szz', 'sxy', 'syz', 'sxz']

        if cat1 == "pressure":
            self.ui.cbx_scat_c2.addItems(res_pres)
        elif cat1 == "u":
            self.ui.cbx_scat_c2.addItems(res_u)
        elif cat1 == "stress":
            self.ui.cbx_scat_c2.addItems(res_str)
        elif cat1 == "estress":
            self.ui.cbx_scat_c2.addItems(res_str)
        elif cat1 == "el. strain":
            self.ui.cbx_scat_c2.addItems(res_str)
        elif cat1 == "pl. strain":
            self.ui.cbx_scat_c2.addItems(res_str)
        elif cat1 == "tot. strain":
            self.ui.cbx_scat_c2.addItems(res_str)
        else:
            self.ui.cbx_scat_c2.addItems(" ")

    def cat_c3_changed(self):
        self.ui.cbx_scat_c3.clear()

        cat1 = self.ui.cbx_cat_c3.currentText()
        res_pres = ['-']
        res_u = ['x', 'y', 'z']
        res_str = ['tot', 's1', 's2', 's3', 'sxx', 'syy', 'szz', 'sxy', 'syz', 'sxz']

        if cat1 == "pressure":
            self.ui.cbx_scat_c3.addItems(res_pres)
        elif cat1 == "u":
            self.ui.cbx_scat_c3.addItems(res_u)
        elif cat1 == "stress":
            self.ui.cbx_scat_c3.addItems(res_str)
        elif cat1 == "estress":
            self.ui.cbx_scat_c3.addItems(res_str)
        elif cat1 == "el. strain":
            self.ui.cbx_scat_c3.addItems(res_str)
        elif cat1 == "pl. strain":
            self.ui.cbx_scat_c3.addItems(res_str)
        elif cat1 == "tot. strain":
            self.ui.cbx_scat_c3.addItems(res_str)
        else:
            self.ui.cbx_scat_c3.addItems(" ")

    def plot_results(self):
        user_cat_c1 = self.ui.cbx_cat_c1.currentText()
        user_cat_c2 = self.ui.cbx_cat_c2.currentText()
        user_cat_c3 = self.ui.cbx_cat_c3.currentText()

        user_subcat_c1 = self.ui.cbx_scat_c1.currentText()
        user_subcat_c2 = self.ui.cbx_scat_c2.currentText()
        user_subcat_c3 = self.ui.cbx_scat_c3.currentText()

        user_cshades = int(self.ui.cshades.value())

        user_zoomx = self.ui.zoom_x.value()
        user_zoomy = self.ui.zoom_y.value()
        user_zoomext = self.ui.zoom_ext.value()
        col_map = 'Greys'
        user_resolution = int(self.ui.resolution.value())

        key_f1 = user_cat_c1 + '-' + user_subcat_c1
        key_f2 = user_cat_c2 + '-' + user_subcat_c2
        key_f3 = user_cat_c3 + '-' + user_subcat_c3

        if self.ui.cbox_zoom.isChecked():
            xi_1 = np.linspace(user_zoomx - 0.5 * user_zoomext, user_zoomx + 0.5 * user_zoomext, user_resolution)
            yi_1 = np.linspace(user_zoomy - 0.5 * user_zoomext, user_zoomy + 0.5 * user_zoomext, user_resolution)
        else:
            xi_1 = np.linspace(-500, 500, user_resolution)
            yi_1 = np.linspace(-500, 500, user_resolution)

        model_comp_current = [self.ui.cbx_comp.currentText()]
        inp_models = files_in_folder(db_folder)

        # Calculate Min-Max Values of all models
        # Get all Min-Max-Values from DB-models
        min_all_1, max_all_1 = grad_db(db_var, inp_models, input_output, key_f1)
        min_all_2, max_all_2 = grad_db(db_var, inp_models, input_output, key_f2)
        min_all_3, max_all_3 = grad_db(db_var, inp_models, input_output, key_f3)
        # Get all Min-Max-Values from Comp-model
        min_i_1, max_i_1 = grad_db(comp_var, model_comp_current, input_output, key_f1)
        min_i_2, max_i_2 = grad_db(comp_var, model_comp_current, input_output, key_f2)
        min_i_3, max_i_3 = grad_db(comp_var, model_comp_current, input_output, key_f3)
        # Select min-max-values for all models in comparison
        minm_1, maxm_1 = grad_all(min_all_1, max_all_1, min_i_1, max_i_1)
        minm_2, maxm_2 = grad_all(min_all_2, max_all_2, min_i_2, max_i_2)
        minm_3, maxm_3 = grad_all(min_all_3, max_all_3, min_i_3, max_i_3)

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

                plt.savefig(folder + i + data_format, bbox_inches='tight')
                plt.close()

                # plt.show()

        pic_creation(db_var, inp_models, fdb_1, data_format, input_output, key_f1, minm_1, maxm_1, user_cshades,
                     col_map, xi_1, yi_1)
        pic_creation(db_var, inp_models, fdb_2, data_format, input_output, key_f2, minm_2, maxm_2, user_cshades,
                     col_map, xi_1, yi_1)
        pic_creation(db_var, inp_models, fdb_3, data_format, input_output, key_f3, minm_3, maxm_3, user_cshades,
                     col_map, xi_1, yi_1)

        pic_creation(comp_var, model_comp_current, fcomp_1, data_format, input_output, key_f1, minm_1, maxm_1,
                     user_cshades, col_map, xi_1, yi_1)
        pic_creation(comp_var, model_comp_current, fcomp_2, data_format, input_output, key_f2, minm_2, maxm_2,
                     user_cshades, col_map, xi_1, yi_1)
        pic_creation(comp_var, model_comp_current, fcomp_3, data_format, input_output, key_f3, minm_3, maxm_3,
                     user_cshades, col_map, xi_1, yi_1)

    def comparison_img(self):

        model_comp_pic_1 = files_in_folder(fcomp_1)
        model_comp_pic_2 = files_in_folder(fcomp_2)
        model_comp_pic_3 = files_in_folder(fcomp_3)

        inp_models_pic_1 = files_in_folder(fdb_1)
        inp_models_pic_2 = files_in_folder(fdb_2)
        inp_models_pic_3 = files_in_folder(fdb_3)

        inp_models_csv = files_in_folder_png(fdb_1)

        for i in model_comp_pic_1:
            path1 = fcomp_1 + i

        for i in model_comp_pic_2:
            path2 = fcomp_2 + i

        for i in model_comp_pic_3:
            path3 = fcomp_3 + i



        def compare_ssim_all(inp_models, db_folder, path):

            list_scores = []
            imageB = cv2.imread(path, cv2.IMREAD_COLOR)

            for i in inp_models:
                # load the two input images
                imageA = cv2.imread(db_folder + i, cv2.IMREAD_COLOR)

                # convert the images to grayscale
                grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

                # compute the Structural Similarity Index (SSIM) between the two
                # images, ensuring that the difference image is returned
                (score, diff) = structural_similarity(grayA, grayB, full=True)
                diff = (diff * 255).astype("uint8")
                #print("SSIM: {}".format(score))

                list_scores.append(score)

            max_ind = list_scores.index(max(list_scores))

            return max_ind

        def plot_max_ssim(max_ind1, max_ind2, max_ind3):

            user_cat_c1 = self.ui.cbx_cat_c1.currentText()
            user_cat_c2 = self.ui.cbx_cat_c2.currentText()
            user_cat_c3 = self.ui.cbx_cat_c3.currentText()

            user_subcat_c1 = self.ui.cbx_scat_c1.currentText()
            user_subcat_c2 = self.ui.cbx_scat_c2.currentText()
            user_subcat_c3 = self.ui.cbx_scat_c3.currentText()

            user_cshades = int(self.ui.cshades.value())

            user_zoomx = self.ui.zoom_x.value()
            user_zoomy = self.ui.zoom_y.value()
            user_zoomext = self.ui.zoom_ext.value()
            user_colormap = str(self.ui.cbx_legend.currentText())

            user_min_c1 = self.ui.min_fig1.value()
            user_min_c2 = self.ui.min_fig2.value()
            user_min_c3 = self.ui.min_fig3.value()

            user_max_c1 = self.ui.max_fig1.value()
            user_max_c2 = self.ui.max_fig2.value()
            user_max_c3 = self.ui.max_fig3.value()

            user_comp_mod = self.ui.cbx_comp.currentText()

            user_resolution = int(self.ui.resolution.value())

            key_f1 = user_cat_c1 + '-' + user_subcat_c1
            key_f2 = user_cat_c2 + '-' + user_subcat_c2
            key_f3 = user_cat_c3 + '-' + user_subcat_c3

            if self.ui.cbox_zoom.isChecked():
                xi_1 = np.linspace(user_zoomx - 0.5 * user_zoomext, user_zoomx + 0.5 * user_zoomext, user_resolution)
                yi_1 = np.linspace(user_zoomy - 0.5 * user_zoomext, user_zoomy + 0.5 * user_zoomext, user_resolution)
            else:
                xi_1 = np.linspace(-500, 500, user_resolution)
                yi_1 = np.linspace(-500, 500, user_resolution)

            model_comp_csv = comp_folder + '/' + user_comp_mod + '.csv'

            model_comp_png1 = fcomp_1 + user_comp_mod + '.png'
            model_comp_png2 = fcomp_2 + user_comp_mod + '.png'
            model_comp_png3 = fcomp_3 + user_comp_mod + '.png'

            inp_csv = files_in_folder_png(fdb_1)

            model_db_csv1 = db_folder + '/' + inp_csv[max_ind1] + '.csv'
            model_db_csv2 = db_folder + '/' + inp_csv[max_ind2] + '.csv'
            model_db_csv3 = db_folder + '/' + inp_csv[max_ind3] + '.csv'

            model_db_png1 = fdb_1 + inp_csv[max_ind1] + '.png'
            model_db_png2 = fdb_2 + inp_csv[max_ind2] + '.png'
            model_db_png3 = fdb_3 + inp_csv[max_ind3] + '.png'

            # Calculations
            results_mod_comp = ans_csvdataimp_complete(model_comp_csv)
            results_mod_db1 = ans_csvdataimp_complete(model_db_csv1)
            results_mod_db2 = ans_csvdataimp_complete(model_db_csv2)
            results_mod_db3 = ans_csvdataimp_complete(model_db_csv3)

            # generate df for plotting
            xval_c1, yval_c1, zval_c1, chead_c1, output_title_c1 = plot_df(results_mod_comp, input_output, key_f1)
            xval_c2, yval_c2, zval_c2, chead_c2, output_title_c2 = plot_df(results_mod_comp, input_output, key_f2)
            xval_c3, yval_c3, zval_c3, chead_c3, output_title_c3 = plot_df(results_mod_comp, input_output, key_f3)

            xval_db1, yval_db1, zval_db1, chead_db1, output_title_db1 = plot_df(results_mod_db1, input_output, key_f1)
            xval_db2, yval_db2, zval_db2, chead_db2, output_title_db2 = plot_df(results_mod_db2, input_output, key_f2)
            xval_db3, yval_db3, zval_db3, chead_db3, output_title_db3 = plot_df(results_mod_db3, input_output, key_f3)

            if self.ui.cbox_man_1.isChecked():
                min_f1 = user_min_c1
                max_f1 = user_max_c1
            else:
                min_f1, max_f1 = graduation(results_mod_comp, results_mod_db1, input_output, key_f1)

            if self.ui.cbox_man_2.isChecked():
                min_f2 = user_min_c2
                max_f2 = user_max_c2
            else:
                min_f2, max_f2 = graduation(results_mod_comp, results_mod_db2, input_output, key_f2)

            if self.ui.cbox_man_3.isChecked():
                min_f3 = user_min_c3
                max_f3 = user_max_c3
            else:
                min_f3, max_f3 = graduation(results_mod_comp, results_mod_db3, input_output, key_f3)

            # define graduation
            user_graduation_f1 = np.linspace(min_f1, max_f1, user_cshades, endpoint=True)
            user_graduation_f2 = np.linspace(min_f2, max_f2, user_cshades, endpoint=True)
            user_graduation_f3 = np.linspace(min_f3, max_f3, user_cshades, endpoint=True)

            # define Colormap
            colm = colormap(user_colormap)

            #### Generation of diff image

            def compare_ssim(path_c, path_db):
                # load the two input images
                imageA = cv2.imread(path_c, cv2.IMREAD_COLOR)
                imageB = cv2.imread(path_db, cv2.IMREAD_COLOR)

                # convert the images to grayscale
                grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

                # compute the Structural Similarity Index (SSIM) between the two
                # images, ensuring that the difference image is returned
                (score, diff) = structural_similarity(grayA, grayB, full=True)
                diff = (diff * 255).astype("uint8")
                #print("SSIM: {}".format(score))

                return diff, score

            diff_1, score1 = compare_ssim(model_comp_png1, model_db_png1)
            diff_2, score2 = compare_ssim(model_comp_png2, model_db_png2)
            diff_3, score3 = compare_ssim(model_comp_png3, model_db_png3)

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = imutils.grab_contours(cnts)

            # loop over the contours
            # for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            # (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # show the output images
            # cv2.imshow('Original', imageA)
            # cv2.imshow(inp_models_pic[max_ind], imageB)
            # cv2.imshow("Diff1", diff_1)
            # cv2.imshow("Diff2", diff_2)
            # cv2.imshow("Diff3", diff_3)
            # cv2.imshow("Thresh", thresh)
            # cv2.waitKey(0)

            # Plotting - general settings

            # figure 1

            # Plotting - general settings
            fig = plt.figure(constrained_layout=True, figsize=(15, 15))

            spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

            # figure 1

            # figure 1

            f1_ax1 = fig.add_subplot(spec[0, 0])
            # Contour plot
            zi_1a = griddata((xval_c1, yval_c1), zval_c1, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f1_1 = plt.contourf(xi_1, yi_1, zi_1a, user_graduation_f1, cmap=colm)

            f1_ax1.axes.get_xaxis().set_visible(False)  # Achsen unsichtbar machen

            plt.ylabel(user_comp_mod)

            plt.title(output_title_c1)

            f1_ax2 = fig.add_subplot(spec[1, 0])
            # Contour plot
            zi_1b = griddata((xval_db1, yval_db1), zval_db1, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f1_2 = plt.contourf(xi_1, yi_1, zi_1b, user_graduation_f1, cmap=colm)

            plt.ylabel(inp_csv[max_ind1])

            fig.colorbar(cf_f1_1, location='bottom')

            # Diffplot

            f1_ax3 = fig.add_subplot(spec[2, 0])
            cf_f1_3 = plt.imshow(diff_1, cmap='Greys_r')
            plt.title("SSIM: {}".format(score1))
            f1_ax3.axes.get_xaxis().set_visible(False)
            f1_ax3.axes.get_yaxis().set_visible(False)

            fig.colorbar(cf_f1_3, location='bottom')

            # figure 2

            f2_ax1 = fig.add_subplot(spec[0, 1])
            zi_2a = griddata((xval_c2, yval_c2), zval_c2, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f2_1 = plt.contourf(xi_1, yi_1, zi_2a, user_graduation_f2, cmap=colm)

            f2_ax1.axes.get_xaxis().set_visible(False)
            # f2_ax1.axes.get_yaxis().set_visible(False)

            plt.title(output_title_c2)

            f2_ax2 = fig.add_subplot(spec[1, 1])
            # Contour plot
            zi_2b = griddata((xval_db2, yval_db2), zval_db2, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f2_2 = plt.contourf(xi_1, yi_1, zi_2b, user_graduation_f2, cmap=colm)

            # f2_ax2.axes.get_yaxis().set_visible(False)
            plt.ylabel(inp_csv[max_ind2])
            fig.colorbar(cf_f2_1, location='bottom')

            # Diffplot

            f2_ax3 = fig.add_subplot(spec[2, 1])
            cf_f2_3 = plt.imshow(diff_2, cmap='Greys_r')
            plt.title("SSIM: {}".format(score2))
            f2_ax3.axes.get_xaxis().set_visible(False)
            f2_ax3.axes.get_yaxis().set_visible(False)

            fig.colorbar(cf_f2_3, location='bottom')

            # figure 3

            f3_ax1 = fig.add_subplot(spec[0, 2])
            zi_3a = griddata((xval_c3, yval_c3), zval_c3, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f3_1 = plt.contourf(xi_1, yi_1, zi_3a, user_graduation_f3, cmap=colm)

            f3_ax1.axes.get_xaxis().set_visible(False)
            # f3_ax1.axes.get_yaxis().set_visible(False)

            plt.title(output_title_c3)

            f3_ax2 = fig.add_subplot(spec[1, 2])
            zi_3b = griddata((xval_db3, yval_db3), zval_db3, (xi_1[None, :], yi_1[:, None]), method='linear')
            cf_f3_2 = plt.contourf(xi_1, yi_1, zi_3b, user_graduation_f3, cmap=colm)

            # f3_ax2.axes.get_yaxis().set_visible(False)
            plt.ylabel(inp_csv[max_ind3])
            fig.colorbar(cf_f3_1, location='bottom')

            # Diffplot

            f3_ax3 = fig.add_subplot(spec[2, 2])
            cf_f3_3 = plt.imshow(diff_3, cmap='Greys_r')
            plt.title("SSIM: {}".format(score1))
            f3_ax3.axes.get_xaxis().set_visible(False)
            f3_ax3.axes.get_yaxis().set_visible(False)

            fig.colorbar(cf_f3_3, location='bottom')

            plt.show()

        max_ind1 = compare_ssim_all(inp_models_pic_1, fdb_1, path1)
        max_ind2 = compare_ssim_all(inp_models_pic_2, fdb_2, path2)
        max_ind3 = compare_ssim_all(inp_models_pic_3, fdb_3, path3)

        plot_max_ssim(max_ind1, max_ind2, max_ind3)


window = MainWindow()

window.show()

sys.exit(app.exec_())
