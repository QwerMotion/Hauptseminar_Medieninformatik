import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Daten kurz laden (nur zum "Lernen" der Kategorien)
df = pd.read_csv('DelayedFlights.csv', usecols=['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Distance', 'DepDelay', 'ArrDelay'])
df = df.dropna(subset=['ArrDelay', 'DepDelay'])

# 2. Preprocessor definieren (exakt wie im Training)
num_features = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'Distance', 'DepDelay']
cat_features = ['UniqueCarrier']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

# 3. Nur "Fitten" (Regeln lernen), NICHT trainieren
preprocessor.fit(df.drop(columns=['ArrDelay']))

# 4. Speichern
joblib.dump(preprocessor, 'preprocessor.joblib')
print("Fertig! Der Dolmetscher wurde gespeichert.")
