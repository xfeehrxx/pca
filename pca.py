import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df_scaled = (df - df.mean()) / df.std()

cov = np.cov(df_scaled.T)


eigvalues, eigvectors = np.linalg.eig(cov)
order = np.argsort(eigvalues)[::-1] 
eigvalues = eigvalues[order]
eigvectors = eigvectors[:,order]



explained_variance = eigvalues / np.sum(eigvalues)

k = 2
pca = np.matmul(df_scaled, eigvectors[:,:k])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca.iloc[:, 0], y=pca.iloc[:, 1])
plt.title('PCA: PC1 vs PC2')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pca.iloc[:, 0], pca.iloc[:, 1], pca.iloc[:, 2])
# ax.set_xlabel('Componente Principal 1')
# ax.set_ylabel('Componente Principal 2')
# ax.set_zlabel('Componente Principal 3')
# plt.title('PCA: PC1 vs PC2 vs PC3')
# plt.show()
print('Encerrado.')
