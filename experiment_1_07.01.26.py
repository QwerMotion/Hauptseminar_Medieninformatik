import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- KONFIGURATION ---
CSV_FILE = 'DelayedFlights.csv'
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
TEST_SIZE = 0.2

# Überprüfen, ob GPU verfügbar ist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training läuft auf: {device}")

# --- 1. DATEN LADEN UND VORBEREITEN ---
print("Lade Daten...")
# Wir laden nur relevante Spalten, um Speicher zu sparen
# Wir ignorieren 'CarrierDelay', 'WeatherDelay' etc., da wir nur ArrDelay vorhersagen wollen.
cols_to_keep = [
    'Month', 'DayofMonth', 'DayOfWeek', 
    'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 
    'Distance', 'ArrDelay'
]

try:
    df = pd.read_csv(CSV_FILE, usecols=cols_to_keep)
except ValueError:
    print("Warnung: Nicht alle Spalten gefunden. Lade alle und filtere manuell.")
    df = pd.read_csv(CSV_FILE)
    df = df[cols_to_keep]

# Datenbereinigung
# 1. Zeilen löschen, wo das ZIEL (ArrDelay) fehlt (wir können nicht trainieren, wenn wir die Lösung nicht kennen)
df = df.dropna(subset=['ArrDelay'])

# Features (X) und Target (y) trennen
X = df.drop(columns=['ArrDelay'])
y = df['ArrDelay'].values.astype(np.float32)

# Feature Engineering / Preprocessing
# Numerische Spalten
num_features = ['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'Distance']
# Kategorische Spalten (Airline)
cat_features = ['UniqueCarrier']

# Wir bauen einen Transformer, der numerische Daten skaliert und kategorische Daten in One-Hot-Vektoren umwandelt
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

print("Bereite Features vor (Skalierung & Encoding)...")
X_processed = preprocessor.fit_transform(X)

# Split in Train und Test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=TEST_SIZE, random_state=42)

# --- 2. PYTORCH DATASET UND DATALOADER ---
class FlightDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1) # Shape (N, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = FlightDataset(X_train, y_train)
test_dataset = FlightDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. DAS NEURALE NETZWERK ---
class DelayPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),       # Gegen Overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)       # Output: Eine einzige Zahl (Minuten Verspätung)
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]
model = DelayPredictor(input_dim).to(device)

# Verlustfunktion und Optimizer
criterion = nn.MSELoss() # Mean Squared Error für Regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAININGS LOOP ---
print("\nStarte Training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()       # Gradienten zurücksetzen
        outputs = model(inputs)     # Vorhersage
        loss = criterion(outputs, labels) # Fehler berechnen
        loss.backward()             # Backpropagation
        optimizer.step()            # Gewichte aktualisieren
        
        running_loss += loss.item()
    
    # Durchschnittlicher Loss pro Epoche
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss (MSE): {running_loss/len(train_loader):.4f}")

# --- 5. EVALUIERUNG / TESTEN ---
print("\nStarte Evaluierung auf Testdaten...")
model.eval() # Wichtig: Deaktiviert Dropout für korrekte Messung
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad(): # Keine Gradientenberechnung nötig beim Testen
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # Speichern für spätere Ansicht
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

avg_mse = test_loss / len(test_loader)
rmse = np.sqrt(avg_mse)

print(f"------------------------------------------------")
print(f"Test Ergebnisse:")
print(f"Mean Squared Error (MSE): {avg_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Durchschnittlicher Fehler in Minuten: ca. +/- {rmse:.2f} Min")
print(f"------------------------------------------------")

# --- BEISPIEL VORHERSAGE ---
print("\nBeispielhafte Vorhersagen vs. Realität (Erste 5 aus dem Testset):")
for i in range(5):
    pred_val = predictions[i][0]
    real_val = actuals[i][0]
    diff = pred_val - real_val
    print(f"Vorhergesagt: {pred_val:6.2f} Min | Echt: {real_val:6.2f} Min | Diff: {diff:6.2f}")

# Speichern des Modells
torch.save(model.state_dict(), "delay_model.pth")
print("\nModell gespeichert als 'delay_model.pth'")
