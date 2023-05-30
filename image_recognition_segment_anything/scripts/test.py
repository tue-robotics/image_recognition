#! /usr/bin/env python

import cv2

import numpy as np
import os.path

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = os.path.expanduser(os.path.join("~", "data", "segment_anything_models", "sam_vit_l_0b3195.pth"))
model_type = "vit_l"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

image = cv2.imread(os.path.expanduser(os.path.join("~", "cabinet_sa.png")))

predictor.set_image(image)

input_box = np.array([400, 30, 500, 160])

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=True,
)

print(f"{masks}")
