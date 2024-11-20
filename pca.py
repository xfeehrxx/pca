import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('mountains_vs_beaches_preferences.csv')
df['Preference'] = df.iloc[:, -1].map({0: 'Praia', 1: 'Montanha'})

text_columns = df.select_dtypes(include=['object']).columns

labels = df[text_columns[-1]] if len(text_columns) > 0 else None

label_encoder = LabelEncoder()
for col in text_columns:
    df[col] = label_encoder.fit_transform(df[col])

df_scaled = (df - df.mean()) / df.std()

cov = np.cov(df_scaled.T)
eigvalues, eigvectors = np.linalg.eig(cov)
order = np.argsort(eigvalues)[::-1]
eigvalues = eigvalues[order]
eigvectors = eigvectors[:, order]

explained_variance = eigvalues / np.sum(eigvalues)

loadings = pd.DataFrame(eigvectors[:, :2],  
                        index=df.columns, 
                        columns=['PC1', 'PC2'])

important_features_pc1 = loadings['PC1'].abs().sort_values(ascending=False)
important_features_pc2 = loadings['PC2'].abs().sort_values(ascending=False)


k = 2
pca = np.matmul(df_scaled, eigvectors[:, :k])

pca_df = pd.DataFrame(pca, columns=['PC1', 'PC2'])
if labels is not None:
    pca_df['Labels'] = labels

top_pc1 = important_features_pc1.index[:2].tolist()
top_pc2 = important_features_pc2.index[:2].tolist()
pc1_text = f"PC1 mais influenciado por: {', '.join(top_pc1)}"
pc2_text = f"PC2 mais influenciado por: {', '.join(top_pc2)}"

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x=pca.iloc[:, 0], y=pca.iloc[:, 1], hue='Labels', palette='Set2')
plt.title('PCA: PC1 vs PC2')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.text(0.05, 0.95, pc1_text, fontsize=10, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.05, 0.90, pc2_text, fontsize=10, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.5))

plt.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print('Encerrado.')
