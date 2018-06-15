# DataScience: EA10 - Instance Based Methods - KNN Exercise
Exercise from the course Data Science at Beuth University of Applied Sciences

### Author
 -----------
* **Name**<br />Dominic Schiller<br />
* **University**<br />Brandenburg University of Applied Sciences<br />
* **E-Mail**<br />dominic.schiller@th-brandenburg.de

---------
### Tasks
> 1. ***(optional)***<br />
> Implement KNN by hand for just 2 dimensions with normalization.<br />
> This is easy because:<br />
>   *  funct: You normalize your data in another table
>  *  funct: You code a simple euclid distance function
>  *  funct: You take a point and calculate the distand to all points
>  *  funct: You take the list from above and sort it
>  *  funct: You aggregate by target variable
>  *  funct: you take the max to determine the targe class<br /><br />
>  
>  you are finished!<br />
>  Note: This is the only chance to implement a machine learning algorithm by hand and hence learn something from the ground up!<br /><br />
>  2. In the logistic regression example, I gave you a new iris data:<br />
>  ***`4.8, 2.5, 5.3, 2.4`***<br />
>  Please classify this flower using KNN.
>  <br /> Here is a good scikit example you can use:
>  
>  *  [Reference [1]](https://www.python-course.eu/k_nearest_neighbor_classifier.php)
>  *  [Packed as notebook](https://drive.google.com/open?id=1DnD_RRAZuanLlJSCmJjRbGtuloZVOirX)

-----
### Solutions
#### Task 1
Please find my hand-implemented KNN classifier in the Python file [***`KNNClassifier.py`***](https://github.com/dominicSchiller/DataScience_EA10_InstanceBasedMethods_KNN/blob/develop/thb/datascience/ibm/KNNClassifier.py) from the Python-module *`thb.datascience.ibm`*<br />
The usage of the custom KNN classifier is demonstrated on the Jupyter notebook [***`Custom_KNN.ipynb`***](https://github.com/dominicSchiller/DataScience_EA10_InstanceBasedMethods_KNN/blob/develop/Custom_KNN.ipynb)