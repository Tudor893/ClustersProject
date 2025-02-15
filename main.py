import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.metrics import silhouette_samples, silhouette_score


df_indicatori = pd.read_csv("data_in/Indicatori.csv", index_col=0)
df_popLoc = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)
indici = list(df_indicatori)[:]

#Cerinta1
medie_tara = df_indicatori["CFA"].mean()
cerinta1 = df_indicatori[df_indicatori["CFA"] > medie_tara]
cerinta1.sort_values(by="CFA", ascending=False).to_csv("data_out/Cerinta1.csv")


#Cerinta2
def fc(t:pd.Series):
    total_indicatori = t[indici].sum()
    total_pop = t["Populatie"].sum()
    return pd.Series(total_indicatori/total_pop*1000, indici)

df_merged = df_indicatori.merge(df_popLoc[["Judet","Populatie"]], left_index=True, right_index=True)
df_grouped = df_merged.groupby(by="Judet").apply(func=fc, include_groups=False)
df_grouped.to_csv("data_out/Cerinta2.csv")


#B

df = pd.read_csv("data_in/LocationQ.csv", index_col=0)

def clean_data(t:pd.DataFrame):
    for column in t.columns:
        if t[column].isna().any():
            if is_numeric_dtype(t[column]):
                t.fillna({column: t[column].mean()}, inplace=True)
            else:
                t.fillna({column: t[column].mode()[0]}, inplace=True)


clean_data(df)

Z = linkage(df.values, method="ward")
linkage_matrix = pd.DataFrame(
    data=Z,
    columns=["Cluster1", "Cluster2", "Distance", "Instances"]
)
linkage_matrix.to_csv("data_out/Linkage_Matrix.csv")

dendrogram(Z)
plt.show()

clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clustering.fit(df)
labels = clustering.labels_
df["Label"] = labels + 1
df.to_csv("data_out/Label_DataFrame.csv")

sil_score = silhouette_score(df, labels)
sil_samples = silhouette_samples(df, labels)

plt.plot(range(1, len(sil_samples) + 1), sil_samples)
plt.show()

for column in df.columns:
    seaborn.histplot(df, x=column, hue="Label", palette='rainbow')
    plt.title(f"Histogram for {column}")
    plt.show()