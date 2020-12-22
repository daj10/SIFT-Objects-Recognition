import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

f = open("matrice_confusion", 'rb')

file = open("classes.cl", 'rb')
data = pickle.load(f)

da = pickle.load(file)
classes = da
a = list(range(len(classes)))

df_cm = pd.DataFrame(data, index=classes,
                     columns=classes)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, xticklabels=True, yticklabels=True)
matplotlib.rc('xtick', labelsize=3)
matplotlib.rc('ytick', labelsize=3)
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=5, rotation=0)
plt.show()