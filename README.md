<img src='https://github.com/gqfiddler/terrorism-attribution-engine/blob/master/map-text.png'>


The <a href="http://www.start.umd.edu/gtd/">Global Terrorism Database</a> is a database of all 180,691 reliably recorded acts of terror since 1970. The data has been collected, cleaned, and curated with the help of various grants and currently resides at the University of Maryland. Roughly 46% of these acts of terrorism that are currently unattributed (perpetrator unknown).

This problem presents a number of challenges:
- the dataset is large, many-featured and strewn with nulls (requires extensive cleaning)
- the data is mixed-type with some categorical and even unstructured data
- the training set of attributed acts differs from the real, untestable test set of unattributed data
- there are 3536 distinct categorical values of the target feature (perpetrating group)

This last constraint is particularly important, affecting on both the accuracy and the computational feasibility of common classification methods.

# The project
In this project (full notebook in the <a href="https://github.com/gqfiddler/terrorism-attribution-engine/blob/master/terrorism_attribution_engine.ipynb">terrorism_atribution_engine.ipynb</a> file above), I walk through the cleaning, visualization, and modeling process over the dataset.  I experiment with a variety of modeling techniques, including returning multiple tiered predictions and utilizing hierarchical schemes.  The purpose here is to show the entire train of thought and the results of each modeling approach, not simply to show the top result (which is somewhat subjective here).

# Results
While the hierarchical schemes show conceptual promise for future development, the most practical model developed here for most applications is the tiered probability output using a decision tree as the base model.  This returns multiple predictions with the associated probability for each, like so:
<img src='https://github.com/gqfiddler/terrorism-attribution-engine/blob/master/prob_predictions.png'>

The absolute (1.0) predictions cover about two-thirds of the labeled data and are 94% accurate.  For non-absolute cases (the remaining one-third), the three most likely predictions contain the correct group 75% of the time.  These results are fairly strong for a 3,500 class problem.

The results and further avenues of research are discussed further in the conclusion section of the notebook.

N.B. The data is not included in this repository because of size constraints, but it can be found at the <a href="http://www.start.umd.edu/gtd/">Global Terrorism Database homepage</a>




