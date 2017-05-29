##   ROC curve 

More reading: [Receiver operating characteristic (Wikipedia)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

The ROC curve is a graphical representation of the contrast between true positive rates and the false positive rate at various thresholds. It’s often used as a proxy for the trade-off between the sensitivity of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).



### AUC

The AUC of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example, i.e. P(score(x+)>score(x−))



### Sensitivity And Specificity

Sensitivity refers to the test's ability to correctly detect patients who do have the condition.In the example of a medical test used to identify a disease, the sensitivity of the test is the proportion of people who test positive for the disease among those who have the disease. Mathematically, this can be expressed as:

![Sensitivity](https://wikimedia.org/api/rest_v1/media/math/render/svg/fbad73213a4578685fefa43ec96ce53533057e11)

Specificity relates to the test's ability to correctly detect patients without a condition.  Consider the example of a medical test for diagnosing a disease. Specificity of a test is the proportion of healthy patients known not to have the disease, who will test negative for it. Mathematically, this can also be written as:


![Specificity](https://wikimedia.org/api/rest_v1/media/math/render/svg/d7856a809dafad4fa9566eef65b37bedeaa53132)

![Image of High Sensitivity Low Specificity](https://upload.wikimedia.org/wikipedia/commons/e/e2/HighSensitivity_LowSpecificity_1401x1050.png)

Lesser False Negatives and more False Positives. Detect more people with the disease.


[Worked out example image](https://en.wikipedia.org/wiki/Template:SensSpecPPVNPV)


## Precision and Recall.

More Reading: [precision and recall-Scikit Learn](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)


[Precision and Recall](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)!

Precision (P) is defined as the number of true positives (Tp) over the number of true positives plus the number of false positives (Fp).

![Precision =](https://wikimedia.org/api/rest_v1/media/math/render/svg/26106935459abe7c266f7b1ebfa2a824b334c807)

Recall (R) is defined as the number of true positives (Tp}) over the number of true positives plus the number of false negatives (Fn).

![Recall =](https://wikimedia.org/api/rest_v1/media/math/render/svg/4c233366865312bc99c832d1475e152c5074891b)

These quantities are also related to the (F1) score, which is defined as the harmonic mean of precision and recall.


