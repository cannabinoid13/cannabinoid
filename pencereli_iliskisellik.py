import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mplcursors
import torch
import torch.nn as nn
import torch.optim as optim

# Neural Relational Inference (NRI) Model
class NRIModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NRIModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# DBSCAN benzeri kümeleme için NRI kullanarak modeli eğitme
def train_nri(X, epochs=10000, lr=0.0001, hidden_dim=64000):
    input_dim = X.shape[1]  # Giriş boyutu, veri özellik sayısına eşittir
    output_dim = 1  # NRI çıktısı, iki nokta arasındaki ilişkiyi ifade eden bir skor olabilir

    model = NRIModel(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, torch.zeros_like(output))  # Basit bir kayıp fonksiyonu, modelin öğrenmesini sağlar
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model

# Veri örneği
data = """
Dronabinol(Delta-9-THC)	0.952	0.995	1.811
(-)-Cannabidiol(CBD)	0.436	0.921	2.149
Dronabinol(Delta-9-THC)	0.827	0.978	1.875
Dronabinol(Delta-9-THC)	0.233	0.997	2.604
(-)-Cannabidiol(CBD)	0.947	0.878	1.395
Dronabinol(Delta-9-THC)	0.321	0.998	2.533
(-)-Cannabidiol(CBD)	0.419	0.941	2.126
Dronabinol(Delta-9-THC)	0.237	0.996	2.637
Dronabinol(Delta-9-THC)	1.061	0.945	1.545
Dronabinol(Delta-9-THC)	0.254	0.998	2.622
(-)-Cannabidiol(CBD)	0.432	0.936	2.118
Dronabinol(Delta-9-THC)	0.877	0.907	1.476
(-)-Cannabidiol(CBD)	0.718	0.451	1.155
Dronabinol(Delta-9-THC)	0.259	0.995	2.592
(-)-Cannabidiol(CBD)	0.325	0.927	1.86
Dronabinol(Delta-9-THC)	0.238	0.999	2.612
(-)-Cannabidiol(CBD)	0.393	0.988	2.019
Cannabigerol(CBG)	0.932	0.699	1.356
Dronabinol(Delta-9-THC)	0.274	0.998	2.568
(-)-Cannabidiol(CBD)	0.95	0.885	1.399
Dronabinol(Delta-9-THC)	0.25	0.996	2.592
Cannabigerol(CBG)	0.37	0.487	1.482
(-)-Cannabidiol(CBD)	1.212	0.462	0.825
Dronabinol(Delta-9-THC)	0.243	0.997	2.639
(-)-Cannabidiol(CBD)	0.915	0.697	1.413
(-)-Cannabidiol(CBD)	0.974	0.279	0.858
Dronabinol(Delta-9-THC)	1.084	0.061	0.543
Dronabinol(Delta-9-THC)	0.249	0.998	2.602
(-)-Cannabidiol(CBD)	0.444	0.928	2.127
(-)-Cannabidiol(CBD)	0.459	0.939	1.908
Dronabinol(Delta-9-THC)	1.105	0.959	1.525
Cannabigerol(CBG)	0.423	0.5	1.454
Dronabinol(Delta-9-THC)	0.823	0.982	1.843
(-)-Cannabidiol(CBD)	1.019	0.858	1.32
Dronabinol(Delta-9-THC)	1.072	0.944	1.573
Dronabinol(Delta-9-THC)	0.186	0.999	2.689
(-)-Cannabidiol(CBD)	1.941	0.895	0.834
Dronabinol(Delta-9-THC)	0.909	0.96	1.657
(-)-Cannabidiol(CBD)	0.996	0.281	0.859
Dronabinol(Delta-9-THC)	0.847	0.988	1.848
Cannabigerol(CBG)	0.956	0.665	1.231
Dronabinol(Delta-9-THC)	0.248	0.997	2.596
Dronabinol(Delta-9-THC)	0.426	0.436	1.38
(-)-Cannabidiol(CBD)	0.469	0.946	1.916
Dronabinol(Delta-9-THC)	1.106	0.955	1.524
Cannabigerol(CBG)	1.008	0.678	1.256
Dronabinol(Delta-9-THC)	0.227	0.998	2.612
(-)-Cannabidiol(CBD)	0.883	0.729	1.507
Dronabinol(Delta-9-THC)	0.169	0.994	2.665
Dronabinol(Delta-9-THC)	0.255	0.997	2.583
(-)-Cannabidiol(CBD)	0.886	0.796	1.421
Dronabinol(Delta-9-THC)	0.248	0.994	2.605
(-)-Cannabidiol(CBD)	1.009	0.248	0.796
Dronabinol(Delta-9-THC)	0.338	0.937	1.963
Cannabigerol(CBG)	0.353	0.479	1.488
(-)-Cannabidiol(CBD)	0.749	0.286	0.919
Dronabinol(Delta-9-THC)	0.3	0.936	1.979
Dronabinol(Delta-9-THC)	1.062	0.949	1.542
Dronabinol(Delta-9-THC)	1.117	0.949	1.508
(-)-Cannabidiol(CBD)	0.949	0.872	1.381
(-)-Cannabidiol(CBD)	0.44	0.931	2.14
Dronabinol(Delta-9-THC)	0.39	0.94	1.948
Dronabinol(Delta-9-THC)	0.224	0.996	2.604
Dronabinol(Delta-9-THC)	0.251	0.993	2.595
(-)-Cannabidiol(CBD)	0.663	0.582	1.408
(-)-Cannabidiol(CBD)	0.465	0.93	2.102
Dronabinol(Delta-9-THC)	1.085	0.95	1.536
Cannabigerol(CBG)	0.367	0.492	1.483
Dronabinol(Delta-9-THC)	0.908	0.953	1.697
(-)-Cannabidiol(CBD)	0.903	0.703	1.45
Dronabinol(Delta-9-THC)	0.325	0.947	1.99
(-)-Cannabidiol(CBD)	0.924	0.706	1.41
Dronabinol(Delta-9-THC)	1.113	0.935	1.466
Cannabigerol(CBG)	1.063	0.758	1.263
Dronabinol(Delta-9-THC)	0.236	0.996	2.616
(-)-Cannabidiol(CBD)	0.42	0.937	2.162
Dronabinol(Delta-9-THC)	0.333	0.994	2.526
(-)-Cannabidiol(CBD)	0.439	0.924	2.018
Dronabinol(Delta-9-THC)	0.306	0.935	1.992
(-)-Cannabidiol(CBD)	0.887	0.695	1.443
Dronabinol(Delta-9-THC)	0.249	0.994	2.594
(-)-Cannabidiol(CBD)	1.225	0.451	0.805
Dronabinol(Delta-9-THC)	0.834	0.987	1.913
(-)-Cannabidiol(CBD)	0.417	0.356	1.454
Dronabinol(Delta-9-THC)	0.167	0.999	2.683
(-)-Cannabidiol(CBD)	0.842	0.58	1.288
Dronabinol(Delta-9-THC)	0.205	0.996	2.658
(-)-Cannabidiol(CBD)	1.077	0.693	1.163
Dronabinol(Delta-9-THC)	1.065	0.951	1.548
(-)-Cannabidiol(CBD)	0.714	0.548	1.333
(-)-Cannabidiol(CBD)	0.443	0.936	2.122
Dronabinol(Delta-9-THC)	0.332	0.949	1.984
Dronabinol(Delta-9-THC)	0.184	0.999	2.653
(-)-Cannabidiol(CBD)	0.925	0.707	1.46
Dronabinol(Delta-9-THC)	0.211	0.998	2.655
Cannabigerol(CBG)	0.35	0.477	1.471
Dronabinol(Delta-9-THC)	0.35	0.949	1.978
(-)-Cannabidiol(CBD)	1.119	0.694	1.163
Dronabinol(Delta-9-THC)	0.225	0.999	2.33
Cannabigerol(CBG)	1.075	0.88	1.407
Dronabinol(Delta-9-THC)	0.845	0.93	1.588
(-)-Cannabidiol(CBD)	0.81	0.79	1.466
Dronabinol(Delta-9-THC)	0.227	0.997	2.617
(-)-Cannabidiol(CBD)	0.452	0.936	2.13
Cannabigerol(CBG)	0.371	0.508	1.519
Dronabinol(Delta-9-THC)	1.032	0.937	1.373
(-)-Cannabidiol(CBD)	0.851	0.916	1.602
Dronabinol(Delta-9-THC)	0.573	0.967	2.119
(-)-Cannabidiol(CBD)	1.155	0.771	1.189
Dronabinol(Delta-9-THC)	1.074	0.983	1.582
(-)-Cannabidiol(CBD)	1.59	0.692	0.731
Dronabinol(Delta-9-THC)	0.524	0.976	2.04
Dronabinol(Delta-9-THC)	0.375	0.911	1.911
(-)-Cannabidiol(CBD)	0.244	0.915	1.955
Dronabinol(Delta-9-THC)	0.485	0.903	1.826
(-)-Cannabidiol(CBD)	0.236	0.988	2.036
(-)-Cannabidiol(CBD)	0.823	0.966	1.713
Dronabinol(Delta-9-THC)	0.627	0.945	2.03
(-)-Cannabidiol(CBD)	0.249	0.92	1.979
(-)-Cannabidiol(CBD)	0.298	0.93	2.216
Dronabinol(Delta-9-THC)	0.871	0.948	1.636
Dronabinol(Delta-9-THC)	0.51	0.893	1.8
(-)-Cannabidiol(CBD)	1.257	0.681	1.017
Dronabinol(Delta-9-THC)	0.689	0.942	1.96
(-)-Cannabidiol(CBD)	0.804	0.798	1.49
Dronabinol(Delta-9-THC)	0.681	0.953	1.979
(-)-Cannabidiol(CBD)	0.93	0.739	1.454
Dronabinol(Delta-9-THC)	0.382	0.905	1.903
(-)-Cannabidiol(CBD)	0.748	0.809	1.523
Dronabinol(Delta-9-THC)	0.413	0.916	1.886
(-)-Cannabidiol(CBD)	1.218	0.662	1.04
Dronabinol(Delta-9-THC)	1.094	0.95	1.524
(-)-Cannabidiol(CBD)	0.908	0.903	1.516
(-)-Cannabidiol(CBD)	0.723	0.838	1.558
(-)-Cannabidiol(CBD)	1.631	0.669	0.704
Dronabinol(Delta-9-THC)	0.663	0.958	2.004
(-)-Cannabidiol(CBD)	0.287	0.902	1.934
Dronabinol(Delta-9-THC)	0.277	0.934	2.032
(-)-Cannabidiol(CBD)	1.074	0.895	1.333
Cannabigerol(CBG)	0.992	0.683	1.17
Dronabinol(Delta-9-THC)	0.187	0.909	1.984
(-)-Cannabidiol(CBD)	0.929	0.739	1.449
(-)-Cannabidiol(CBD)	0.914	0.68	1.3
Dronabinol(Delta-9-THC)	0.119	0.99	2.249
(-)-Cannabidiol(CBD)	0.776	0.81	1.501
Dronabinol(Delta-9-THC)	0.366	0.886	1.805
(-)-Cannabidiol(CBD)	0.792	0.804	1.491
Dronabinol(Delta-9-THC)	0.686	0.977	1.914
Dronabinol(Delta-9-THC)	0.503	0.894	1.797
(-)-Cannabidiol(CBD)	1.55	0.65	0.682
Dronabinol(Delta-9-THC)	0.945	0.889	1.586
(-)-Cannabidiol(CBD)	0.764	0.76	1.433
Dronabinol(Delta-9-THC)	0.957	0.96	1.607
(-)-Cannabidiol(CBD)	1.038	0.934	1.495
Dronabinol(Delta-9-THC)	0.523	0.893	1.782
(-)-Cannabidiol(CBD)	1.231	0.594	0.889
(-)-Cannabidiol(CBD)	0.78	0.816	1.508
Dronabinol(Delta-9-THC)	0.623	0.52	1.31
Dronabinol(Delta-9-THC)	0.631	0.501	1.416
(-)-Cannabidiol(CBD)	0.804	0.607	1.385
Dronabinol(Delta-9-THC)	0.629	0.966	2.051
(-)-Cannabidiol(CBD)	0.916	0.755	1.479
(-)-Cannabidiol(CBD)	0.732	0.829	1.564
Dronabinol(Delta-9-THC)	0.74	0.98	2.017
(-)-Cannabidiol(CBD)	0.902	0.666	1.297
(-)-Cannabidiol(CBD)	0.295	0.983	2.014
Dronabinol(Delta-9-THC)	0.512	0.955	1.96
(-)-Cannabidiol(CBD)	0.78	0.815	1.499
(-)-Cannabidiol(CBD)	0.28	0.892	1.926
Dronabinol(Delta-9-THC)	0.939	0.865	1.565
Dronabinol(Delta-9-THC)	0.852	0.928	1.628
(-)-Cannabidiol(CBD)	1.079	0.675	1.082
Dronabinol(Delta-9-THC)	0.968	0.792	1.399
Cannabigerol(CBG)	1.074	0.771	1.358
(-)-Cannabidiol(CBD)	1.381	0.758	1.002
(-)-Cannabidiol(CBD)	0.937	0.708	1.314
(-)-Cannabidiol(CBD)	0.828	0.84	1.448
Dronabinol(Delta-9-THC)	0.902	0.901	1.671
(-)-Cannabidiol(CBD)	0.802	0.936	1.663
Dronabinol(Delta-9-THC)	0.391	0.861	1.727
(-)-Cannabidiol(CBD)	0.961	0.755	1.435
Dronabinol(Delta-9-THC)	0.024	0.998	2.296
(-)-Cannabidiol(CBD)	0.746	0.932	1.79
Dronabinol(Delta-9-THC)	0.866	0.939	1.583
(-)-Cannabidiol(CBD)	1.57	0.632	0.638
Dronabinol(Delta-9-THC)	0.549	0.968	2.146
(-)-Cannabidiol(CBD)	0.904	0.756	1.489
Dronabinol(Delta-9-THC)	0.36	0.893	1.834
(-)-Cannabidiol(CBD)	0.955	0.753	1.436
Dronabinol(Delta-9-THC)	0.645	0.961	2.028
(-)-Cannabidiol(CBD)	0.825	0.945	1.678
(-)-Cannabidiol(CBD)	1.503	0.662	0.746
(-)-Cannabidiol(CBD)	1.584	0.695	0.761
(-)-Cannabidiol(CBD)	0.242	0.919	1.967
Dronabinol(Delta-9-THC)	0.927	0.837	1.532
(-)-Cannabidiol(CBD)	0.783	0.778	1.468
Dronabinol(Delta-9-THC)	0.197	0.949	2.024
(-)-Cannabidiol(CBD)	1.038	0.894	1.44
(-)-Cannabidiol(CBD)	0.314	0.913	1.924
Dronabinol(Delta-9-THC)	0.549	0.886	1.753
(-)-Cannabidiol(CBD)	1.305	0.681	0.969
Dronabinol(Delta-9-THC)	0.495	0.97	2.015
(-)-Cannabidiol(CBD)	0.748	0.946	1.795
(-)-Cannabidiol(CBD)	0.23	0.986	2.058
Dronabinol(Delta-9-THC)	1.056	0.848	1.374
(-)-Cannabidiol(CBD)	0.698	0.866	1.643
Dronabinol(Delta-9-THC)	0.881	0.939	1.671
(-)-Cannabidiol(CBD)	1.347	0.814	0.991
Dronabinol(Delta-9-THC)	0.403	0.968	2.332
(-)-Cannabidiol(CBD)	0.918	0.728	1.445
(-)-Cannabidiol(CBD)	0.152	0.988	2.059
Dronabinol(Delta-9-THC)	0.874	0.911	1.682
Dronabinol(Delta-9-THC)	0.642	0.969	2.102
(-)-Cannabidiol(CBD)	1.361	0.788	1.125
(-)-Cannabidiol(CBD)	0.291	0.983	2.26
Dronabinol(Delta-9-THC)	0.553	0.515	1.545
(-)-Cannabidiol(CBD)	0.93	0.711	1.33
Dronabinol(Delta-9-THC)	0.442	0.885	1.839
(-)-Cannabidiol(CBD)	1.152	0.93	1.373
Dronabinol(Delta-9-THC)	0.457	0.916	1.874
(-)-Cannabidiol(CBD)	0.744	0.831	1.535
(-)-Cannabidiol(CBD)	1.04	0.747	1.308
Dronabinol(Delta-9-THC)	0.635	0.945	1.831
(-)-Cannabidiol(CBD)	1.352	0.664	0.912
Dronabinol(Delta-9-THC)	0.333	0.933	1.957
(-)-Cannabidiol(CBD)	0.774	0.951	1.772
(-)-Cannabidiol(CBD)	0.882	0.657	1.299
Dronabinol(Delta-9-THC)	1.133	0.947	1.392
(-)-Cannabidiol(CBD)	0.913	0.662	1.292
(-)-Cannabidiol(CBD)	0.896	0.664	1.289
(-)-Cannabidiol(CBD)	0.293	0.789	1.982
Dronabinol(Delta-9-THC)	0.638	0.949	1.832
Dronabinol(Delta-9-THC)	0.403	0.862	1.773
(-)-Cannabidiol(CBD)	0.96	0.767	1.444
(-)-Cannabidiol(CBD)	1.328	0.661	0.932
(-)-Cannabidiol(CBD)	0.792	0.79	1.468
Dronabinol(Delta-9-THC)	0.687	0.511	1.263
(-)-Cannabidiol(CBD)	0.385	0.922	2.134
Dronabinol(Delta-9-THC)	0.656	0.957	2.016
(-)-Cannabidiol(CBD)	1.521	0.652	0.718
(-)-Cannabidiol(CBD)	0.791	0.799	1.478
Dronabinol(Delta-9-THC)	0.572	0.965	2.177
(-)-Cannabidiol(CBD)	0.91	0.679	1.313
(-)-Cannabidiol(CBD)	1.232	0.736	1.092
(-)-Cannabidiol(CBD)	0.285	0.897	1.938
(-)-Cannabidiol(CBD)	0.78	0.802	1.491
(-)-Cannabidiol(CBD)	0.414	0.917	2.106
(-)-Cannabidiol(CBD)	0.287	0.916	1.972
Dronabinol(Delta-9-THC)	0.364	0.907	1.857
(-)-Cannabidiol(CBD)	0.802	0.636	1.338
Dronabinol(Delta-9-THC)	0.043	0.995	2.287
(-)-Cannabidiol(CBD)	1.208	0.589	0.886
Dronabinol(Delta-9-THC)	0.03	0.999	2.306
(-)-Cannabidiol(CBD)	0.926	0.741	1.448
(-)-Cannabidiol(CBD)	1.622	0.667	0.686
(-)-Cannabidiol(CBD)	1.523	0.668	0.73
(-)-Cannabidiol(CBD)	0.966	0.725	1.314
Dronabinol(Delta-9-THC)	0.815	0.94	1.671
(-)-Cannabidiol(CBD)	0.982	0.757	1.418
(-)-Cannabidiol(CBD)	0.401	0.922	2.122
Dronabinol(Delta-9-THC)	0.541	0.895	1.771
Dronabinol(Delta-9-THC)	0.673	0.978	2.083
Dronabinol(Delta-9-THC)	0.687	0.959	1.988
(-)-Cannabidiol(CBD)	0.312	0.902	1.919
(-)-Cannabidiol(CBD)	0.778	0.31	0.89
Dronabinol(Delta-9-THC)	1.054	0.905	1.521
(-)-Cannabidiol(CBD)	0.285	0.908	1.948
Dronabinol(Delta-9-THC)	0.538	0.892	1.767
(-)-Cannabidiol(CBD)	0.292	0.933	1.96
(-)-Cannabidiol(CBD)	0.273	0.913	1.953
(-)-Cannabidiol(CBD)	0.34	0.942	2.187
Dronabinol(Delta-9-THC)	0.543	0.978	2.173
Cannabigerol(CBG)	0.744	0.461	1.143
Dronabinol(Delta-9-THC)	0.692	0.967	1.968
(-)-Cannabidiol(CBD)	1.463	0.741	0.934
Cannabigerol(CBG)	1.179	0.288	0.619
Dronabinol(Delta-9-THC)	0.859	0.916	1.618
(-)-Cannabidiol(CBD)	1.172	0.673	1.102
"""  # Verilerin tamamını buraya yapıştırın

# Verileri düzenleme ve uygun formata dönüştürme
lines = data.strip().split("\n")
titles = [line.split()[0] for line in lines]
values = [[float(val) for val in line.split()[1:]] for line in lines]
X = np.array(values)

# NRI modelini eğitme
model = train_nri(torch.tensor(X).float())

# Eğitilmiş modeli kullanarak örnekler arasındaki ilişkileri tahmin etme
relationships = model(torch.tensor(X).float())

# Kümeleme yapmak için örnekler arasındaki ilişkileri kullanma
# Örnek olarak, eşik değeri kullanarak ilişkileri bir kümeleme algoritması ile etiketleyebilirsiniz.

# Dördüncü boyutu renk olarak temsil ederek 3D scatter plot oluşturma
colors = relationships.squeeze().tolist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# CBD olan noktaları ve onların ilişkiselliğini belirleme
cbd_indices = [i for i, title in enumerate(titles) if "CBD" in title]
cbd_colors = [color for i, color in enumerate(colors) if i in cbd_indices]

sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap='plasma', alpha=0.5)
sc_cbd = ax.scatter(X[cbd_indices, 0], X[cbd_indices, 1], X[cbd_indices, 2], c=cbd_colors, cmap='Reds', alpha=0.5)

# Renk skalası
cbar = plt.colorbar(sc)
cbar.set_label('Relationship Score')

# İnteraktif etiketler
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f'{titles[sel.target.index]}: {X[sel.target.index, 3]}'))

ax.set_title('NRI Clustering')
ax.set_xlabel('Align Score')
ax.set_ylabel('Vector Score')
ax.set_zlabel('Fitness Score')

# Kümeleme sonuçlarını ayrı bir pencerede gösterme
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
ax_inset.scatter(X[:, 0], X[:, 1], c=colors, cmap='plasma', alpha=0.5)
ax_inset.scatter(X[cbd_indices, 0], X[cbd_indices, 1], c='red', alpha=0.5)  # CBD olanları kırmızı olarak göster
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_title('Clusters')

# Grafik çizme işlemleri
plt.savefig('NRI_Clustering.png', dpi=300)

plt.show()
