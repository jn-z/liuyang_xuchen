import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import math
from caculate_H import calculate_H

class Fractal():

    def __init__(self):
         self.s_list = [4,6,8,10,12,15]
        # self.s_list =

    method of image graying
    def gray(self, src):

        gray_img = np.uint8(src[:, :, 0] * 0.144 + src[:, :, 1] * 0.587 + src[:, :, 2] * 0.299)

        # cla_img = cv2.bilateralFilter(gray_img, 3, 64, 64)

        # #clahe processing
        # clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize = (32, 32))
        # cla_img = clahe.apply(bil_img)

        return gray_img

    #  differential box dimension counting (DBC)
    def differential_box_counting_and_multiplying_dimesion(self, gray_img):

        h, w = gray_img.shape[:2]
        M = min(h, w)
        N_L_list = []
        n_l_row_col_list = []
        n_l_row_col_list_final = []
        G = gray_img.max()
        for s in self.s_list:  # the box side length: 2 ~ M//2
            H = G*s/M  # high of the box
            N_L = 0  # initialization of the box number
            n_l_row_col_list = []
            for row in range(h // s):  # h//s: the number of rows in the box;  w//s: the number of columns in the box
                for col in range(w // s):
                    temp = gray_img[row * s:(row + 1) * s, col * s:(col + 1) * s]
                    if np.all(temp.data) == 0:
                        continue
                    n_L_row_col = math.ceil((np.max(gray_img[row * s:(row + 1) * s, col * s:(col + 1) * s]) - np.min(
                        gray_img[row * s:(row + 1) * s, col * s:(col + 1) * s])) / H + 1)
                    n_l_row_col_list.append(n_L_row_col)
                    N_L += n_L_row_col
            n_l_row_col_list = [element/N_L for element in n_l_row_col_list if N_L != 0]
            n_l_row_col_list_final.append(n_l_row_col_list)
            N_L_list.append(N_L)

        return N_L_list, M, n_l_row_col_list_final

    def least_squares(self, x, y):

        """
        (1) input datesets of x and y
        (2) the straight line is fitted by Least-square method
        (3) output a coefficient(w), intercept(b) and coefficient of determination (r)
        (4) the fitting straight line : y = wx + b
        """
        # 计算y的标准偏差
        std_dev = np.std(y)

        # 定义离群点的阈值（比如两倍的标准偏差加上均值）
        threshold = np.mean(y) + 2 * std_dev

        # 找出并剔除离群点
        inliers_mask = y <= threshold
        inliers_x = x[inliers_mask]
        inliers_y = y[inliers_mask]

        x = inliers_x
        y = inliers_y

        x_ = x.mean()
        y_ = y.mean()

        m1 = np.zeros(1)
        m2 = np.zeros(1)
        m3 = np.zeros(1)

        k1 = np.zeros(1)
        k2 = np.zeros(1)
        k3 = np.zeros(1)

        for i in np.arange(len(x)):
            m1 += (x[i] - x_) * y[i]
            m2 += np.square(x[i])
            m3 += x[i]

            k1 += (x[i] - x_) * (y[i] - y_)
            k2 += np.square(x[i] - x_)
            k3 += np.square(y[i] - y_)

        w = m1 / (m2 - 1 / len(x) * np.square(m3))
        b = y_ - w * x_
        R = k1 / np.sqrt(k2 * k3)  # r is the coefficient of determination , dedicating fitting

        return w, b, R

    def plot_line(self, x, y, w, b, R):

        # print(w, b, r ** 2)
        y_pred = w * x + b

        # create a fig and an axes
        fig, ax = plt.subplots(figsize=(10, 5))

        # fontsyle: SimHei(黑体)，support chinese
        plt.rcParams['font.sans-serif'] = ['SimHei']

        ax.plot(x, y, 'co', markersize=6, label='scatter datas')
        ax.plot(x, y_pred, 'r-', linewidth=2, label='y = %.4fx + %.4f' % (w, b))

        # set xlim and yxlim
        ax.set_aspect("0.5")
        # ax.set_xlim(1, 6)
        # ax.set_ylim(2, 12)

        # set x_ticks and y_ticks

        ax.tick_params(labelsize=16)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # create grids
        ax.grid(which="major", axis="both")

        # display labels
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
        ax.legend(prop=font1)

        # set x_label and y_label
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        ax.set_xlabel("ln 1/r", fontdict=font2)
        ax.set_ylabel("ln Nr", fontdict=font2)

        # set a position of "R^2"
        ax.text(-5.0, 14.8, 'R^2 = %.4f' % float(R * R), fontdict=font1,
                verticalalignment='center', horizontalalignment='left', rotation=0)

        plt.savefig("line.jpg", dpi=300, bbox_inches="tight")

        plt.show()

    def execute(self, img):


        # M is the size of the image
        N_L_list, M, n_l_row_col_list_final = self.differential_box_counting_and_multiplying_dimesion(img)
        L_num = [round(M / s) for s in self.s_list]
        x1 = np.log([round(M / s) for s in self.s_list])
        y1 = np.log(N_L_list)

        # fitting a straight line
        w1, b1, r1 = self.least_squares(x1, y1)

        # self.plot_line(x1, y1, w1, b1, r1)

        x2 = np.array([(s / M) for s in self.s_list])
        row_num = len(n_l_row_col_list_final)
        md = []
        list_row_q = []
        for q in range(25,26):
            for i in range(row_num):
                row = n_l_row_col_list_final[i]
                row_q = np.log(1 / sum(element**q for element in row))
                list_row_q.append(row_q)
            w2, b2, r2 = self.least_squares(np.log(1 / x2), np.array(list_row_q))

            w2 = w2 / (q-1)
            md.append(b2)
        self.plot_line(np.log(1 / x2), np.array(list_row_q), w2, b2, r2)
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.use('Qt5Agg')
        plt.figure()
        plt.scatter(np.log(1 / x2), np.array(list_row_q))
        plt.title('Two Column Vectors')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

        plt.figure()
        plt.scatter(range(2,50),md)
        plt.show()

        return b1, md, r1, r2, w1, w2




