import matplotlib.pyplot as plt
import itertools
import numpy as np

from sklearn.metrics import roc_curve, confusion_matrix


def plot_confusion_matrix(labels, classifications_ground_truth, classifications_predicted_label):
    """
    Plot the confusion matrix of a given classification result
    :param labels: Input labels
    :param classifications_ground_truth: Ground truth labels
    :param classifications_predicted_label: Predicted labels
    """
    plt.figure()

    cnf_matrix = confusion_matrix(classifications_ground_truth, classifications_predicted_label, labels)

    # Plot normalized confusion matrix
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=80)
    plt.yticks(tick_marks, labels)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black",
                 fontsize=6)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid()


def plot_false_positive_true_positive_rates(labels, classifications_ground_truth_as_score_matrix,
                                            classifications_scores):
    """
    Plot the false positive true positive rates per label
    :param labels: Input labels
    :param classifications_ground_truth_as_score_matrix: Zero score matrix with ones on the ground truth
    :param classifications_scores: The classification scores per label
    """

    # setup consistent colors + markers for each label
    colors_per_label = dict(zip(labels, [plt.get_cmap('gist_rainbow')(i) for i in np.linspace(0, 1, len(labels))]))
    markers_per_label = dict(zip(labels, itertools.cycle([' ', 'o', 'x'])))

    plt.figure()
    plt.rc("axes", labelsize=15)

    # Compute ROC curve and ROC area for each class
    false_positive_rate_per_label = {}
    true_positive_rate_per_label = {}
    unknown_thresholds_per_label = {}
    for i, label in enumerate(labels):
        false_positive_rate_per_label[label], true_positive_rate_per_label[label], unknown_thresholds_per_label[
            label] = \
            roc_curve(classifications_ground_truth_as_score_matrix[:, i], classifications_scores[:, i])

    for label in labels:
        plt.plot(unknown_thresholds_per_label[label], true_positive_rate_per_label[label],
                 color=colors_per_label[label], marker=markers_per_label[label], label='Tpr {}'.format(label))
        plt.plot(unknown_thresholds_per_label[label], false_positive_rate_per_label[label],
                 color=colors_per_label[label], marker=markers_per_label[label], linestyle='dashed',
                 label='Fpr {}'.format(label))

    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel(r'True  $\frac{T_p(t)}{T_p(t) + F_p(t)}$ & False $\frac{F_p(t)}{F_p(t) + T_n(t)}$ Positive Rate: ')
    plt.xlabel('Threshold ($t$)')
    plt.title('Threshold vs. True & False positive rate')
