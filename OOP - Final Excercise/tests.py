import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath


if __name__ == '__main__':
    figure, axes = plt.subplots()
    Drawing_colored_circle = mpatches.Circle((0.6, 0.6), 0.2, ec='blue', fc='red')

    axes.set_aspect(1)
    axes.add_artist(Drawing_colored_circle)
    plt.show()

