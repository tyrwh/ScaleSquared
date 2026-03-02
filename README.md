# ScaleSquared
This is a catch-all repo for models and code related to Breeding Insight's segmentation projects on adult trout spawners.

These were part of a set of projects with Ken Overturf, formerly the USDA-ARS trout breeder in Hagerman, Idaho, and his team. The overall goal was to take useful measurements of adult fish from video, including:
* Size measurements like total length or standard length
* More complex morphometrics that may be good predictors of fillet weight or fillet yield
* Metrics of growth deformities like curved spines

The first step was to simply delineate the boundaries of the fish to get total area as viewed from the side. We did this with a [YOLO11](https://github.com/ultralytics/ultralytics) detection model, the bounding boxes of which were fed into [SAM2](github.com/facebookresearch/sam2) to segment the fish.

After that, we trained a first-draft YOLO11 segmentation model to segment the following features:
- Head
- Pectoral fin
- Dorsal fin
- Pelvic fin
- Anal fin
- Adipose fin
- Caudal fin

These projects were tabled indefinitely in 2025 when Ken left the USDA. As of early 2026, they are still tabled.

## Datasets

The images used in these projects are all frames pulled from ~110 videos taken by Ken Overturf and his team in 2024.

For the detection model, I pulled ~330 frames from each video and used grounded DINO labeling on [Roboflow](https://app.roboflow.com) with the prompt "large brown fish." I manually checked all images to verify that the bounding box was drawn correctly and fixed any mistakes.

For the segmentation model, I picked 10 images from different videos and labeled the features by hand using grounded SAM2 on Roboflow. The model was trained with only 8 images as the training set.

Full datasets are available, but we should probably get the written permission of Ken Overturf and/or the relevant person currently at the USDA before transferring those in their entirety.

## Models

There are two model files, `yolo11m_body_det.pt` and `yolo11m_full_seg.pt`. Both are based on the YOLO11m base model for detection and segmentation, respectively.

`yolo11m_body.pt` is based on the YOLO11m detection base model.

`yolo11m_full_seg.pt` is a segmentation model.

## Questions

Please contact Tyr Wiesner-Hanks ([twiesnerhanks@ufl.edu](mailto:twiesnerhanks@ufl.edu)) with any questions.



