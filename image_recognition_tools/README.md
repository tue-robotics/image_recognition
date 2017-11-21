image_recognition_tools
=======================

Tool(s) for image recognition

Classifier_metrics
------------------

Usage: 
```bash
rosrun image_recognition_tools classifier_metrics -i results.csv
```

The classifier_metrics script produces a plot that gives a good view of the performance of a given classifier. As input, it takes a .csv file of the following format:
- First row are column headers
- There is a ground_truth column 
