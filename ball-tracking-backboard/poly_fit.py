import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.image as mpimg


class PolyFit:
    def __init__(self):
        pass

    @staticmethod
    def fit(coef_x, coef_y, degree):
        lin = LinearRegression()
        lin.fit(coef_x, coef_y)

        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(coef_x)

        poly.fit(x_poly, coef_y)
        lin2 = LinearRegression()
        lin2.fit(x_poly, coef_y)

        # plt.scatter(coef_x, coef_y, color='blue')
        #
        # plt.plot(coef_x, lin.predict(coef_x), color='red')
        # plt.title('Linear Regression')
        #
        # plt.show()

        plt.scatter(coef_x, coef_y, color='blue')

        plt.plot(coef_x, lin2.predict(poly.fit_transform(coef_x)), color='red')
        # plt.title('Polynomial Regression')

        # fig = plt.figure()
        # plt.savefig("my_img.png")
        # shot = plt.gca()
        # line = shot.lines[0]
        # print(len(line.get_xdata()))

        # draw the figure first...
        # fig.canvas.draw()

        # Now we can save it to a numpy array.
        # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # print(data)
        plt.show()

    @staticmethod
    def fitNP(x, y, degree):
        z = np.polyfit(x, y, degree)
        # print(z)

        p = np.poly1d(z)
        # print(p(5))
        # print(p)

        xp = np.linspace(x.min(), x.max(), 100)
        # print(xp)
        image = mpimg.imread("detected_0.jpg")
        plt.imshow(image)

        plt.plot(x, y, '.', xp, p(xp), '-')
        #plt.gca().invert_yaxis()
        plt.savefig("curve.jpg")


