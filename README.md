# The problem

The Global Terrorism Database is a database of all 180,691 reliably recorded acts of terror since 1970. The data has been collected, cleaned, and curated with the help of various grants and currently resides at the University of Maryland. In this project, I'll be trying to attribute responsibility for the 46% of these acts of terrorism that are currently unattributed.

This problem presents a number of challenges:

the dataset is large, many-featured and strewn with nulls (requires extensive cleaning)
the data is mixed-type with some categorical and even unstructured data
the training set of attributed acts differs from the real, untestable test set of unattributed data
But there's one problem in particular that will define this project: there are 3536 distinct categorical values of the target feature. This places constraints on both our accuracy and our computational feasibility.

