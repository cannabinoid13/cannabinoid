# Cannabinoid Synergistic Effect Analysis

This repository contains two Python files in total, one that clusters with the DBSCAN algorithm using only two data (and one title) (DBSCAN.py) and one that analyses the relationship between three data sets (and one title) with NRI (xxxxx). The xxxxxx file also performs clustering in parallel with NRI.

With these codes, it is analysed how the Align Score, Vector Score and Fitness Score data obtained as a result of processing the frames obtained from molecular dynamics simulations with 3D pharmacophore modelling method, how the relationship between the frames is established and how they are clustered. 

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

For neural relational inference (NRI) analysis, you need to use three data sets (Align Score, Vector Score and Fitness Score). The created model will learn the relationship network between the elements in the three data sets within the framework of the determined parameters and will give the result output. You can change the learning parameters according to your computer hardware. Using epochs=10000, lr=0.0001, hidden_dim=64000 codes as they are will put extra load on your processor. You can use the codes as they are to make a strong analysis of the relationality.


```
pencereli_iliskisellik.py

```

As a result, you will get a graph showing the placement of the Align Score, Vector Score and Fitness Score data in three-dimensional space in a three-dimensional graph. The fourth dimension of this graph is a scale showing the relationality analysed with NRI and the data is coloured according to this scale in the three-dimensional graph.

### Reference
We thank the official implementation of neural relational inference at
https://github.com/ethanfetaya/NRI

Neural relational inference to learn long-range allosteric interactions in proteins from molecular dynamics simulations.**  
Jingxuan Zhu,  Juexin Wang, Weiwei Han,  Dong Xu,
Nature communications 13, no. 1 (2022): 1-16.
https://www.nature.com/articles/s41467-022-29331-3

Erich Schubert, Jörg Sander, Martin Ester, Hans Peter Kriegel, and Xiaowei Xu. 2017. DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN. ACM Trans. Database Syst. 42, 3, Article 19 (September 2017), 21 pages. https://doi.org/10.1145/3068335



