# cannabinoid
Cannabinoid NRI

With these codes, it is useful to analyse how the Align Score, Vector Score and Fitness Score data obtained as a result of processing the frames obtained from molecular dynamics simulations with 3D pharmacophore modelling method establishes a relationship between the frames.

Data Example: The data example contains the data points and their titles. Each data point consists of a position in three-dimensional space and a fourth dimension represented by a score.

Data Processing: The data is represented as a list, where each element contains the title and four feature values. This data is then converted into an appropriate numpy array. Training the NRI Model: The train_nri function is used to train the data with the NRI model. The model attempts to learn the relationships between the data points. Creating the Plot: A 3D scatter plot is used to visualize the data. Each point represents an example in the dataset. The color of the points represents the relational scores predicted by the NRI model. Additionally, points containing CBD are emphasized with different shades of color. Displaying Clustering Results in a Separate Window: Using the inset_axes function, a small window is added below the main scatter plot. This window provides a more detailed view of the clustering results and better visualizes the relational scores of the points on the main graph. This code learns the relationships between examples in the dataset using the NRI model and uses matplotlib library to visualize and analyze them. It also creates an additional window on the graph to show clustering results in more detail.


![DBSCAN_NRI_toplu](https://github.com/cannabinoid13/cannabinoid/assets/166438571/0050c557-74e8-4ffb-98d8-1d6dba5fb00f)

### Requirements
Python 3.7

### Hardware
At least intel i5 processor (12th generation)

### Before Using Codes (Preparation Phase)

After obtaining the Trajactory windows, 3D pharmacophore modelling is performed for each frame with a suitable software to obtain Align Score, Vector Score and Fitness Score data.

### Use of Codes

After obtaining Align Score, Vector Score and Fitness Score data, enter Align Score and Fitness Score into DBSACN.py (or any two data types for which clustering status is desired)

```
DBSCAN.py
```

In this way, you will graphically obtain a visual representation of the clusters between two data types using the DBSCAN algorithm.

For neural relational inference (NRI) analysis, you need to use three data sets (Align Score, Vector Score and Fitness Score). The created model will learn the relationship network between the elements in the three data sets within the framework of the determined parameters and will give the result output.


```
pencereli_iliskisellik.py

```



