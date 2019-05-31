#!/usr/bin/env python

from __future__ import print_function

from image_recognition_tensorflow.object_recognizer import ObjectRecognizer
from image_recognition_util.classification_score_matrix import ClassificationScoreMatrix
from image_recognition_util.image_reader import read_annotated


def evaluate_classifier(graph_path, labels_path, annotated_dir, output,
                        input_tensor="Cast:0", output_tensor="final_result:0"):
    """
    Evaluate a given classifier against a directory of annotated images.
    The script will output a .csv that can be evaluated.
    :param graph_path: Path to graph.pb
    :param labels_path: Path to labels.txt
    :param annotated_dir: Directory path where the annotated images are stored, to be verified
    :param output: Output file
    :param input_tensor: Input tensor name
    :param output_tensor: Output tensor name
    :return: Final accuracy
    """
    object_recognizer = ObjectRecognizer(graph_path, labels_path, input_tensor, output_tensor)

    classification_score_matrix = ClassificationScoreMatrix(object_recognizer.labels)

    print("Evaluating classified with labels: {}".format(object_recognizer.labels))

    matches = []
    for label, image, filename in read_annotated(annotated_dir):
        if label not in object_recognizer.labels:
            print("Skipping image with label '{}', label not present in classifier".format(label))
            continue

        scores = object_recognizer.classify(image)
        match, best_label, best_score = classification_score_matrix.add_classification(label, scores)

        matches.append(match)
        print("{}({}/{})\t{}\t{} classified as {} ({:.3f})\033[0m".format("\033[92m" if match else "\033[91m",
                                                                          matches.count(True), len(matches),
                                                                          filename, label, best_label, best_score))

    classification_score_matrix.write_to_file(output)

    accuracy = float(matches.count(True)) / len(matches)
    print("Final accuracy: {}".format(accuracy))
    return accuracy
