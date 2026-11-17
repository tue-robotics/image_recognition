#!/usr/bin/env python
import os

from image_recognition_tensorflow.evaluate_classifier import evaluate_classifier
from image_recognition_tensorflow.retrain import defaults, main


def test_retrain_and_evaluate():
    pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(pkg_path, '/tmp/models')
    assets_path = os.path.join(pkg_path, 'test/assets')

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    main(os.path.join(assets_path, "training"), models_path, models_path, defaults.steps, defaults.batch)
    accuracy = evaluate_classifier(os.path.join(models_path, "output_graph.pb"),
                                   os.path.join(models_path, "output_labels.txt"),
                                   os.path.join(assets_path, "verification"), "/tmp/result.csv")
    assert accuracy > 0.5


if __name__ == "__main__":
    test_retrain_and_evaluate()
