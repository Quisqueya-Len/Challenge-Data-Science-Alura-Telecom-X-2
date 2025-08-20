import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

url = 'https://raw.githubusercontent.com/ximec74/TelecomX_Challenge_parte2/refs/heads/main/df_churn.csv'
datos = pd.read_csv(url)

datos = datos.drop(columns=[c for c in ['customerID','cuentas_diarias','TipoServicio'] if c in datos.columns])
datos = datos.dropna(subset=["account.Charges.Total"])

# Heatmap (codificar y correlación)
df_corr = datos.copy()
le = LabelEncoder()
for col in df_corr.select_dtypes(include='object').columns:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))
df_corr.fillna(df_corr.median(numeric_only=True), inplace=True)
matriz_correlacion = df_corr.corr()
plt.figure(figsize=(12,12))
sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, square=True)
plt.title("Matriz de Correlación entre Variables", fontsize=12)
plt.tight_layout()
plt.savefig('Matriz_correlación_entre_variables.png', dpi=120)
plt.show()

# Agrupar "No internet service" -> "No"
cols_to_fix = ['internet.OnlineSecurity','internet.OnlineBackup','internet.DeviceProtection',
               'internet.TechSupport','internet.StreamingTV','internet.StreamingMovies']
for col in cols_to_fix:
    if col in datos.columns:
        datos[col] = datos[col].replace('No internet service','No')

# Feature encoding binario
mappings = {
    'Churn': {'Yes':0,'No':1},
    'customer.gender': {'Male':0,'Female':1},
    'customer.Partner': {'Yes':0,'No':1},
    'customer.Dependents': {'Yes':0,'No':1},
    'phone.PhoneService': {'Yes':0,'No':1},
    'account.PaperlessBilling': {'Yes':0,'No':1},
    'internet.OnlineSecurity': {'Yes':0,'No':1},
    'internet.OnlineBackup': {'Yes':0,'No':1},
    'internet.DeviceProtection': {'Yes':0,'No':1},
    'internet.TechSupport': {'Yes':0,'No':1},
    'internet.StreamingTV': {'Yes':0,'No':1},
    'internet.StreamingMovies': {'Yes':0,'No':1}
}
for col, m in mappings.items():
    if col in datos.columns:
        datos[col] = datos[col].replace(m).astype(int if set(m.values())<=set([0,1]) else object)

# One-Hot Encoding para variables categóricas
categoricas = [c for c in ['phone.MultipleLines','internet.InternetService','account.Contract','account.PaymentMethod'] if c in datos.columns]
one_hot_enc = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categoricas), remainder='passthrough')
datos_enc = one_hot_enc.fit_transform(datos)
cols_enc = one_hot_enc.get_feature_names_out()
datos_transformados = pd.DataFrame(datos_enc, columns=cols_enc)
datos_transformados.info()

# Pie chart proporción de churn
valores = datos_transformados['remainder__Churn'].value_counts()
etiquetas = ['Clientes que permanecen','Clientes que abadonan']
plt.figure(figsize=(4,4))
plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=90, colors=['skyblue','goldenrod'], labeldistance=0.1)
plt.title('Proporción de Evasión vs Permanencia\nClientes con contratos de hasta 6 años', fontsize=11)
plt.axis('equal')
plt.savefig('Clientes_Proporcion_Evasion_Permanencia.png', dpi=120)
plt.show()

# Análisis dirigido: tenure vs churn
mean_tenure_group = datos.groupby("Churn")["customer.tenure"].mean()
mean_tenure_global = datos["customer.tenure"].mean()
fig, axes = plt.subplots(1,2, figsize=(9,4))
sns.boxplot(data=datos, x="Churn", y="customer.tenure", ax=axes[0], palette="Set2")
axes[0].set_title("Boxplot Tenure vs Churn")
axes[0].axhline(mean_tenure_global, color="red", linewidth=2, label=f"Global {mean_tenure_global:.1f}")
for cat, avg in mean_tenure_group.items():
    axes[0].axhline(avg, color="blue", linestyle="--", linewidth=1, label=f"{cat}: {avg:.1f}")
axes[0].legend()
sns.histplot(data=datos, y="customer.tenure", hue="Churn", multiple="stack", ax=axes[1], palette="Set2")
axes[1].set_title("Distribución de Antigüedad por Churn")
axes[1].axhline(mean_tenure_global, color="red", linewidth=2, label=f"Global {mean_tenure_global:.1f}")
for cat, avg in mean_tenure_group.items():
    axes[1].axhline(avg, color="blue", linestyle="--", linewidth=1, label=f"{cat}: {avg:.1f}")
axes[1].legend()
plt.tight_layout()
plt.show()

# Análisis dirigido: charges.total vs churn
mean_charges_group = datos.groupby("Churn")["account.Charges.Total"].mean()
mean_charges_global = datos["account.Charges.Total"].mean()
fig, axes = plt.subplots(1,2, figsize=(9,4))
sns.boxplot(data=datos, x="Churn", y="account.Charges.Total", ax=axes[0], palette="Set2")
axes[0].set_title("Boxplot Charges.Total vs Churn")
axes[0].axhline(mean_charges_global, color="red", linewidth=2, label=f"Global {mean_charges_global:.0f}")
for cat, avg in mean_charges_group.items():
    axes[0].axhline(avg, color="blue", linestyle="--", linewidth=1, label=f"{cat}: {avg:.0f}")
axes[0].legend()
sns.histplot(data=datos, y="account.Charges.Total", hue="Churn", multiple="stack", ax=axes[1], palette="Set2")
axes[1].set_title("Distribución de Charges.Total por Churn")
axes[1].axhline(mean_charges_global, color="red", linewidth=2, label=f"Global {mean_charges_global:.0f}")
for cat, avg in mean_charges_group.items():
    axes[1].axhline(avg, color="blue", linestyle="--", linewidth=1, label=f"{cat}: {avg:.0f}")
axes[1].legend()
plt.tight_layout()
plt.show()
