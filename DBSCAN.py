import numpy as np
import matplotlib.pyplot as plt
import mplcursors


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def dbscan(X, eps, min_samples):
    clusters = []
    visited = set()

    for i, x in enumerate(X):
        if i in visited:
            continue
        visited.add(i)
        neighbors = find_neighbors(X, x, eps)
        if len(neighbors) < min_samples:
            clusters.append([i])  # Her nokta kendi kümesine dahil edilir
        else:
            cluster = expand_cluster(X, i, neighbors, eps, min_samples, visited)
            clusters.append(cluster)

    return clusters


def find_neighbors(X, x, eps):
    neighbors = []
    for i, xi in enumerate(X):
        if euclidean_distance(x, xi) < eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(X, i, neighbors, eps, min_samples, visited):
    cluster = [i]
    for neighbor in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            neighbor_neighbors = find_neighbors(X, X[neighbor], eps)
            if len(neighbor_neighbors) >= min_samples:
                neighbors.extend(neighbor_neighbors)
        if neighbor not in cluster:
            cluster.append(neighbor)
    return cluster


# Verileri düzenleme
data = """
Dronabinol(Delta-9-THC)	0.952	1.811
(-)-Cannabidiol(CBD)	0.436	2.149
Dronabinol(Delta-9-THC)	0.827	1.875
Dronabinol(Delta-9-THC)	0.233	2.604
(-)-Cannabidiol(CBD)	0.947	1.395
Dronabinol(Delta-9-THC)	0.321	2.533
(-)-Cannabidiol(CBD)	0.419	2.126
Dronabinol(Delta-9-THC)	0.237	2.637
Dronabinol(Delta-9-THC)	1.061	1.545
Dronabinol(Delta-9-THC)	0.254	2.622
(-)-Cannabidiol(CBD)	0.432	2.118
Dronabinol(Delta-9-THC)	0.877	1.476
(-)-Cannabidiol(CBD)	0.718	1.155
Dronabinol(Delta-9-THC)	0.259	2.592
(-)-Cannabidiol(CBD)	0.325	1.86
Dronabinol(Delta-9-THC)	0.238	2.612
(-)-Cannabidiol(CBD)	0.393	2.019
Cannabigerol(CBG)	0.932	1.356
Dronabinol(Delta-9-THC)	0.274	2.568
(-)-Cannabidiol(CBD)	0.95	1.399
Dronabinol(Delta-9-THC)	0.25	2.592
Cannabigerol(CBG)	0.37	1.482
(-)-Cannabidiol(CBD)	1.212	0.825
Dronabinol(Delta-9-THC)	0.243	2.639
(-)-Cannabidiol(CBD)	0.915	1.413
(-)-Cannabidiol(CBD)	0.974	0.858
Dronabinol(Delta-9-THC)	1.084	0.543
Dronabinol(Delta-9-THC)	0.249	2.602
(-)-Cannabidiol(CBD)	0.444	2.127
(-)-Cannabidiol(CBD)	0.459	1.908
Dronabinol(Delta-9-THC)	1.105	1.525
Cannabigerol(CBG)	0.423	1.454
Dronabinol(Delta-9-THC)	0.823	1.843
(-)-Cannabidiol(CBD)	1.019	1.32
Dronabinol(Delta-9-THC)	1.072	1.573
Dronabinol(Delta-9-THC)	0.186	2.689
(-)-Cannabidiol(CBD)	1.941	0.834
Dronabinol(Delta-9-THC)	0.909	1.657
(-)-Cannabidiol(CBD)	0.996	0.859
Dronabinol(Delta-9-THC)	0.847	1.848
Cannabigerol(CBG)	0.956	1.231
Dronabinol(Delta-9-THC)	0.248	2.596
Dronabinol(Delta-9-THC)	0.426	1.38
(-)-Cannabidiol(CBD)	0.469	1.916
Dronabinol(Delta-9-THC)	1.106	1.524
Cannabigerol(CBG)	1.008	1.256
Dronabinol(Delta-9-THC)	0.227	2.612
(-)-Cannabidiol(CBD)	0.883	1.507
Dronabinol(Delta-9-THC)	0.169	2.665
Dronabinol(Delta-9-THC)	0.255	2.583
(-)-Cannabidiol(CBD)	0.886	1.421
Dronabinol(Delta-9-THC)	0.248	2.605
(-)-Cannabidiol(CBD)	1.009	0.796
Dronabinol(Delta-9-THC)	0.338	1.963
Cannabigerol(CBG)	0.353	1.488
(-)-Cannabidiol(CBD)	0.749	0.919
Dronabinol(Delta-9-THC)	0.3	1.979
Dronabinol(Delta-9-THC)	1.062	1.542
Dronabinol(Delta-9-THC)	1.117	1.508
(-)-Cannabidiol(CBD)	0.949	1.381
(-)-Cannabidiol(CBD)	0.44	2.14
Dronabinol(Delta-9-THC)	0.39	1.948
Dronabinol(Delta-9-THC)	0.224	2.604
Dronabinol(Delta-9-THC)	0.251	2.595
(-)-Cannabidiol(CBD)	0.663	1.408
(-)-Cannabidiol(CBD)	0.465	2.102
Dronabinol(Delta-9-THC)	1.085	1.536
Cannabigerol(CBG)	0.367	1.483
Dronabinol(Delta-9-THC)	0.908	1.697
(-)-Cannabidiol(CBD)	0.903	1.45
Dronabinol(Delta-9-THC)	0.325	1.99
(-)-Cannabidiol(CBD)	0.924	1.41
Dronabinol(Delta-9-THC)	1.113	1.466
Cannabigerol(CBG)	1.063	1.263
Dronabinol(Delta-9-THC)	0.236	2.616
(-)-Cannabidiol(CBD)	0.42	2.162
Dronabinol(Delta-9-THC)	0.333	2.526
(-)-Cannabidiol(CBD)	0.439	2.018
Dronabinol(Delta-9-THC)	0.306	1.992
(-)-Cannabidiol(CBD)	0.887	1.443
Dronabinol(Delta-9-THC)	0.249	2.594
(-)-Cannabidiol(CBD)	1.225	0.805
Dronabinol(Delta-9-THC)	0.834	1.913
(-)-Cannabidiol(CBD)	0.417	1.454
Dronabinol(Delta-9-THC)	0.167	2.683
(-)-Cannabidiol(CBD)	0.842	1.288
Dronabinol(Delta-9-THC)	0.205	2.658
(-)-Cannabidiol(CBD)	1.077	1.163
Dronabinol(Delta-9-THC)	1.065	1.548
(-)-Cannabidiol(CBD)	0.714	1.333
(-)-Cannabidiol(CBD)	0.443	2.122
Dronabinol(Delta-9-THC)	0.332	1.984
Dronabinol(Delta-9-THC)	0.184	2.653
(-)-Cannabidiol(CBD)	0.925	1.46
Dronabinol(Delta-9-THC)	0.211	2.655
Cannabigerol(CBG)	0.35	1.471
Dronabinol(Delta-9-THC)	0.35	1.978
(-)-Cannabidiol(CBD)	1.119	1.163
Dronabinol(Delta-9-THC)	0.225	2.33
Cannabigerol(CBG)	1.075	1.407
Dronabinol(Delta-9-THC)	0.845	1.588
(-)-Cannabidiol(CBD)	0.81	1.466
Dronabinol(Delta-9-THC)	0.227	2.617
(-)-Cannabidiol(CBD)	0.452	2.13
Cannabigerol(CBG)	0.371	1.519
Dronabinol(Delta-9-THC)	1.032	1.373
(-)-Cannabidiol(CBD)	0.851	1.602
Dronabinol(Delta-9-THC)	0.573	2.119
(-)-Cannabidiol(CBD)	1.155	1.189
Dronabinol(Delta-9-THC)	1.074	1.582
(-)-Cannabidiol(CBD)	1.59	0.731
Dronabinol(Delta-9-THC)	0.524	2.04
Dronabinol(Delta-9-THC)	0.375	1.911
(-)-Cannabidiol(CBD)	0.244	1.955
Dronabinol(Delta-9-THC)	0.485	1.826
(-)-Cannabidiol(CBD)	0.236	2.036
(-)-Cannabidiol(CBD)	0.823	1.713
Dronabinol(Delta-9-THC)	0.627	2.03
(-)-Cannabidiol(CBD)	0.249	1.979
(-)-Cannabidiol(CBD)	0.298	2.216
Dronabinol(Delta-9-THC)	0.871	1.636
Dronabinol(Delta-9-THC)	0.51	1.8
(-)-Cannabidiol(CBD)	1.257	1.017
Dronabinol(Delta-9-THC)	0.689	1.96
(-)-Cannabidiol(CBD)	0.804	1.49
Dronabinol(Delta-9-THC)	0.681	1.979
(-)-Cannabidiol(CBD)	0.93	1.454
Dronabinol(Delta-9-THC)	0.382	1.903
(-)-Cannabidiol(CBD)	0.748	1.523
Dronabinol(Delta-9-THC)	0.413	1.886
(-)-Cannabidiol(CBD)	1.218	1.04
Dronabinol(Delta-9-THC)	1.094	1.524
(-)-Cannabidiol(CBD)	0.908	1.516
(-)-Cannabidiol(CBD)	0.723	1.558
(-)-Cannabidiol(CBD)	1.631	0.704
Dronabinol(Delta-9-THC)	0.663	2.004
(-)-Cannabidiol(CBD)	0.287	1.934
Dronabinol(Delta-9-THC)	0.277	2.032
(-)-Cannabidiol(CBD)	1.074	1.333
Cannabigerol(CBG)	0.992	1.17
Dronabinol(Delta-9-THC)	0.187	1.984
(-)-Cannabidiol(CBD)	0.929	1.449
(-)-Cannabidiol(CBD)	0.914	1.3
Dronabinol(Delta-9-THC)	0.119	2.249
(-)-Cannabidiol(CBD)	0.776	1.501
Dronabinol(Delta-9-THC)	0.366	1.805
(-)-Cannabidiol(CBD)	0.792	1.491
Dronabinol(Delta-9-THC)	0.686	1.914
Dronabinol(Delta-9-THC)	0.503	1.797
(-)-Cannabidiol(CBD)	1.55	0.682
Dronabinol(Delta-9-THC)	0.945	1.586
(-)-Cannabidiol(CBD)	0.764	1.433
Dronabinol(Delta-9-THC)	0.957	1.607
(-)-Cannabidiol(CBD)	1.038	1.495
Dronabinol(Delta-9-THC)	0.523	1.782
(-)-Cannabidiol(CBD)	1.231	0.889
(-)-Cannabidiol(CBD)	0.78	1.508
Dronabinol(Delta-9-THC)	0.623	1.31
Dronabinol(Delta-9-THC)	0.631	1.416
(-)-Cannabidiol(CBD)	0.804	1.385
Dronabinol(Delta-9-THC)	0.629	2.051
(-)-Cannabidiol(CBD)	0.916	1.479
(-)-Cannabidiol(CBD)	0.732	1.564
Dronabinol(Delta-9-THC)	0.74	2.017
(-)-Cannabidiol(CBD)	0.902	1.297
(-)-Cannabidiol(CBD)	0.295	2.014
Dronabinol(Delta-9-THC)	0.512	1.96
(-)-Cannabidiol(CBD)	0.78	1.499
(-)-Cannabidiol(CBD)	0.28	1.926
Dronabinol(Delta-9-THC)	0.939	1.565
Dronabinol(Delta-9-THC)	0.852	1.628
(-)-Cannabidiol(CBD)	1.079	1.082
Dronabinol(Delta-9-THC)	0.968	1.399
Cannabigerol(CBG)	1.074	1.358
(-)-Cannabidiol(CBD)	1.381	1.002
(-)-Cannabidiol(CBD)	0.937	1.314
(-)-Cannabidiol(CBD)	0.828	1.448
Dronabinol(Delta-9-THC)	0.902	1.671
(-)-Cannabidiol(CBD)	0.802	1.663
Dronabinol(Delta-9-THC)	0.391	1.727
(-)-Cannabidiol(CBD)	0.961	1.435
Dronabinol(Delta-9-THC)	0.024	2.296
(-)-Cannabidiol(CBD)	0.746	1.79
Dronabinol(Delta-9-THC)	0.866	1.583
(-)-Cannabidiol(CBD)	1.57	0.638
Dronabinol(Delta-9-THC)	0.549	2.146
(-)-Cannabidiol(CBD)	0.904	1.489
Dronabinol(Delta-9-THC)	0.36	1.834
(-)-Cannabidiol(CBD)	0.955	1.436
Dronabinol(Delta-9-THC)	0.645	2.028
(-)-Cannabidiol(CBD)	0.825	1.678
(-)-Cannabidiol(CBD)	1.503	0.746
(-)-Cannabidiol(CBD)	1.584	0.761
(-)-Cannabidiol(CBD)	0.242	1.967
Dronabinol(Delta-9-THC)	0.927	1.532
(-)-Cannabidiol(CBD)	0.783	1.468
Dronabinol(Delta-9-THC)	0.197	2.024
(-)-Cannabidiol(CBD)	1.038	1.44
(-)-Cannabidiol(CBD)	0.314	1.924
Dronabinol(Delta-9-THC)	0.549	1.753
(-)-Cannabidiol(CBD)	1.305	0.969
Dronabinol(Delta-9-THC)	0.495	2.015
(-)-Cannabidiol(CBD)	0.748	1.795
(-)-Cannabidiol(CBD)	0.23	2.058
Dronabinol(Delta-9-THC)	1.056	1.374
(-)-Cannabidiol(CBD)	0.698	1.643
Dronabinol(Delta-9-THC)	0.881	1.671
(-)-Cannabidiol(CBD)	1.347	0.991
Dronabinol(Delta-9-THC)	0.403	2.332
(-)-Cannabidiol(CBD)	0.918	1.445
(-)-Cannabidiol(CBD)	0.152	2.059
Dronabinol(Delta-9-THC)	0.874	1.682
Dronabinol(Delta-9-THC)	0.642	2.102
(-)-Cannabidiol(CBD)	1.361	1.125
(-)-Cannabidiol(CBD)	0.291	2.26
Dronabinol(Delta-9-THC)	0.553	1.545
(-)-Cannabidiol(CBD)	0.93	1.33
Dronabinol(Delta-9-THC)	0.442	1.839
(-)-Cannabidiol(CBD)	1.152	1.373
Dronabinol(Delta-9-THC)	0.457	1.874
(-)-Cannabidiol(CBD)	0.744	1.535
(-)-Cannabidiol(CBD)	1.04	1.308
Dronabinol(Delta-9-THC)	0.635	1.831
(-)-Cannabidiol(CBD)	1.352	0.912
Dronabinol(Delta-9-THC)	0.333	1.957
(-)-Cannabidiol(CBD)	0.774	1.772
(-)-Cannabidiol(CBD)	0.882	1.299
Dronabinol(Delta-9-THC)	1.133	1.392
(-)-Cannabidiol(CBD)	0.913	1.292
(-)-Cannabidiol(CBD)	0.896	1.289
(-)-Cannabidiol(CBD)	0.293	1.982
Dronabinol(Delta-9-THC)	0.638	1.832
Dronabinol(Delta-9-THC)	0.403	1.773
(-)-Cannabidiol(CBD)	0.96	1.444
(-)-Cannabidiol(CBD)	1.328	0.932
(-)-Cannabidiol(CBD)	0.792	1.468
Dronabinol(Delta-9-THC)	0.687	1.263
(-)-Cannabidiol(CBD)	0.385	2.134
Dronabinol(Delta-9-THC)	0.656	2.016
(-)-Cannabidiol(CBD)	1.521	0.718
(-)-Cannabidiol(CBD)	0.791	1.478
Dronabinol(Delta-9-THC)	0.572	2.177
(-)-Cannabidiol(CBD)	0.91	1.313
(-)-Cannabidiol(CBD)	1.232	1.092
(-)-Cannabidiol(CBD)	0.285	1.938
(-)-Cannabidiol(CBD)	0.78	1.491
(-)-Cannabidiol(CBD)	0.414	2.106
(-)-Cannabidiol(CBD)	0.287	1.972
Dronabinol(Delta-9-THC)	0.364	1.857
(-)-Cannabidiol(CBD)	0.802	1.338
Dronabinol(Delta-9-THC)	0.043	2.287
(-)-Cannabidiol(CBD)	1.208	0.886
Dronabinol(Delta-9-THC)	0.03	2.306
(-)-Cannabidiol(CBD)	0.926	1.448
(-)-Cannabidiol(CBD)	1.622	0.686
(-)-Cannabidiol(CBD)	1.523	0.73
(-)-Cannabidiol(CBD)	0.966	1.314
Dronabinol(Delta-9-THC)	0.815	1.671
(-)-Cannabidiol(CBD)	0.982	1.418
(-)-Cannabidiol(CBD)	0.401	2.122
Dronabinol(Delta-9-THC)	0.541	1.771
Dronabinol(Delta-9-THC)	0.673	2.083
Dronabinol(Delta-9-THC)	0.687	1.988
(-)-Cannabidiol(CBD)	0.312	1.919
(-)-Cannabidiol(CBD)	0.778	0.89
Dronabinol(Delta-9-THC)	1.054	1.521
(-)-Cannabidiol(CBD)	0.285	1.948
Dronabinol(Delta-9-THC)	0.538	1.767
(-)-Cannabidiol(CBD)	0.292	1.96
(-)-Cannabidiol(CBD)	0.273	1.953
(-)-Cannabidiol(CBD)	0.34	2.187
Dronabinol(Delta-9-THC)	0.543	2.173
Cannabigerol(CBG)	0.744	1.143
Dronabinol(Delta-9-THC)	0.692	1.968
(-)-Cannabidiol(CBD)	1.463	0.934
Cannabigerol(CBG)	1.179	0.619
Dronabinol(Delta-9-THC)	0.859	1.618
(-)-Cannabidiol(CBD)	1.172	1.102

"""  # Verilerin tamamını buraya yapıştırın

# Verileri düzenleme ve uygun formata dönüştürme
lines = data.strip().split("\n")
values = [[float(val) for val in line.split()[1:]] for line in lines]

# Verileri ölçekleme
X = np.array(values)

# DBSCAN modeli uygulama
eps = 0.1
min_samples = 2
clusters = dbscan(X, eps, min_samples)

# Renk skalası oluşturma
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan']

# Görselleştirme
fig, ax = plt.subplots(figsize=(8, 6))
for i, cluster in enumerate(clusters):
    if cluster is not None:
        cluster_points = X[cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}', s=50,
                   alpha=0.5)

# İsimleri ekleme
names = [line.split()[0] for line in lines]
for i, name in enumerate(names):
    if name == "(-)-Cannabidiol(CBD)":
        ax.text(X[i, 0], X[i, 1], "CBD", fontsize=8)

# İnteraktif etiketler
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(names[sel.target.index]))

plt.title('DBSCAN Clustering')
plt.xlabel('Align Score')
plt.ylabel('Fitness Skore')
plt.legend()
plt.savefig('DBSCAN.png', dpi=300)
plt.show()
