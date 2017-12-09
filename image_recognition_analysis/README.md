image_recognition_analysis
=======================

Tool(s) for image recognition

Classifier_metrics
------------------

Usage: 
```bash
rosrun image_recognition_tools classifier_metrics results.csv
```

The classifier_metrics script produces a plot that gives a good view of the performance of a given classifier. As input, it takes a .csv file of the following format:
- First row are column headers
- Other rows are 1 per image
- There is a ground_truth column
- For each output label the classifier, there must be a column with the weight of the corresponding class

e.g. 
```csv
ground_truth,cat,dog
cat,0.9,0.1
dog,0.2,0.8
dog,0.3,0.7
```
