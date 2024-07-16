import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import os

def plot_calibration_curves(tag, clf_list, X, y, save_dir):
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 1)
    colors = plt.cm.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:1, :1])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        display = CalibrationDisplay.from_estimator(
            clf,
            X,
            y,
            n_bins=5,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    plt.savefig(os.path.join(save_dir, tag + '_calibration_curves.png'))

    # # Add histogram
    # grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    # for i, (_, name) in enumerate(clf_list):
    #     row, col = grid_positions[i]
    #     ax = fig.add_subplot(gs[row, col])
    #
    #     ax.hist(
    #         calibration_displays[name].y_prob,
    #         range=(0, 1),
    #         bins=10,
    #         label=name,
    #         color=colors(i),
    #     )
    #     ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    # plt.show()