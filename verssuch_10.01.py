import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

# 1. DATEN LADEN
# Ersetze 'deine_datei.csv' mit deinem Dateipfad
# Wir laden nur n zeilen zum Testen, wenn die Datei riesig ist, sonst entferne 'nrows'
df = pd.read_csv('C:/Users/lukas/OneDrive/Desktop/DelayedFlights.csv/DelayedFlights.csv', nrows=100000) 


# 2. VORVERARBEITUNG
input_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Distance', 'DepDelay']
target_col = 'ArrDelay'

# Wichtig: Zeilen entfernen, wo ArrDelay oder DepDelay NaN (leer) sind
df = df.dropna(subset=input_cols + [target_col])

X = df[input_cols]
y = df[target_col]

# Split in Training und Test (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Da 'UniqueCarrier' Text ist, müssen wir ihn umwandeln
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['UniqueCarrier'])
    ],
    remainder='passthrough' # Alle anderen Spalten (Zahlen) einfach durchlassen
)

# 3. MODELL ERSTELLEN (Random Forest)
# n_estimators=100 bedeutet, wir nutzen 100 Bäume
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10))
])

print("Starte Training (das kann kurz dauern)...")
model.fit(X_train, y_train)

# 4. VORHERSAGE & EVALUIERUNG
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Performance Random Forest ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Durchschnittlicher Fehler: +/- {rmse:.2f} Min")

# 5. VISUALISIERUNG EINES BAUMES
# Ein Random Forest besteht aus vielen Bäumen. Wir extrahieren EINEN davon zur Ansicht.
rf_model = model.named_steps['regressor']
single_tree = rf_model.estimators_[0] # Wir nehmen den ersten Baum aus dem Wald

plt.figure(figsize=(20, 10))
plot_tree(single_tree, 
          max_depth=5, # Wir begrenzen die Tiefe für die Lesbarkeit
          feature_names=model.named_steps['preprocessor'].get_feature_names_out(),
          filled=True, 
          fontsize=4)
plt.title("Ausschnitt eines einzelnen Entscheidungsbaums (Tiefe 3)")
plt.show()

# 6. FEATURE IMPORTANCE (Was war dem Modell am wichtigsten?)
# Das ist oft hilfreicher als der Baum selbst
importances = rf_model.feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()

# Sortieren und anzeigen
indices = np.argsort(importances)[::-1]
print("\n--- Top 5 Wichtigste Features ---")
for i in range(5):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")