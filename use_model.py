import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd

# --- 1. MODELL-KLASSE DEFINIEREN (muss identisch zum Training sein) ---
class DelayPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 2. SETUP & LADEN ---
try:
    preprocessor = joblib.load('preprocessor.joblib')
    # Bestimme die Input-Dimension aus dem Preprocessor
    # Wir erstellen ein Dummy-DataFrame, um die transformierte Form zu sehen
    dummy_df = pd.DataFrame(columns=['Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSArrTime', 'UniqueCarrier', 'Distance', 'DepDelay'])
    # Wir brauchen mindestens eine Zeile zum Transformieren für die Form-Prüfung
    input_dim = preprocessor.transform(pd.DataFrame([[1,1,1,1200,1300,'WN',500,10]], columns=dummy_df.columns)).shape[1]
    
    model = DelayPredictor(input_dim)
    model.load_state_dict(torch.load("delay_model_2.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Fehler beim Laden der Dateien: {e}")
    exit()

# --- 3. GUI LOGIK ---
def predict_delay():
    try:
        # Daten aus den Feldern auslesen
        data = {
            'Month': [int(entry_month.get())],
            'DayofMonth': [int(entry_day.get())],
            'DayOfWeek': [int(entry_dow.get())],
            'CRSDepTime': [int(entry_deptime.get())],
            'CRSArrTime': [int(entry_arrtime.get())],
            'UniqueCarrier': [combo_carrier.get()],
            'Distance': [float(entry_dist.get())],
            'DepDelay': [float(entry_dep_delay.get())]
        }
        
        df_input = pd.DataFrame(data)
        
        # Transformation
        processed_input = preprocessor.transform(df_input)
        input_tensor = torch.tensor(processed_input, dtype=torch.float32)
        
        # Vorhersage
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        # Ergebnis anzeigen
        color = "red" if prediction > 15 else "green"
        lbl_result.config(text=f"Vorhergesagte Verspätung: {prediction:.2f} Min", foreground=color)
        
    except Exception as e:
        messagebox.showerror("Fehler", f"Ungültige Eingabe: {e}")

# --- 4. GUI STRUKTUR ---
root = tk.Tk()
root.title("Flug-Verspätungs-Check")
root.geometry("400x500")

style = ttk.Style()
style.configure("TLabel", padding=5)

main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(main_frame, text="Flugdaten eingeben", font=("Arial", 14, "bold")).grid(row=0, columnspan=2, pady=10)

# Eingabefelder
fields = [
    ("Monat (1-12):", "1"),
    ("Tag des Monats:", "15"),
    ("Wochentag (1-7):", "3"),
    ("Abflugzeit (HHMM):", "1430"),
    ("Ankunftszeit (HHMM):", "1645"),
    ("Distanz (Meilen):", "500"),
    ("Abflug-Verspätung (Min):", "10")
]

entries = {}
for i, (label, default) in enumerate(fields):
    ttk.Label(main_frame, text=label).grid(row=i+1, column=0, sticky="w")
    entry = ttk.Entry(main_frame)
    entry.insert(0, default)
    entry.grid(row=i+1, column=1, pady=5)
    entries[label] = entry

# Extra: Dropdown für Carrier
ttk.Label(main_frame, text="Airline Code:").grid(row=8, column=0, sticky="w")
# Hier die wichtigsten US-Carrier als Beispiel, OneHotEncoder handhabt Unbekannte
carriers = ['WN', 'AA', 'MQ', 'UA', 'OO', 'DL', 'XE', 'CO', 'US', 'EV', 'NW', 'FL', 'B6', 'OH', '9E', 'AS', 'YV', 'F9', 'HA', 'AQ']
combo_carrier = ttk.Combobox(main_frame, values=carriers)
combo_carrier.set("UA")
combo_carrier.grid(row=8, column=1, pady=5)

# Variablen-Mapping für die Logik
entry_month = entries["Monat (1-12):"]
entry_day = entries["Tag des Monats:"]
entry_dow = entries["Wochentag (1-7):"]
entry_deptime = entries["Abflugzeit (HHMM):"]
entry_arrtime = entries["Ankunftszeit (HHMM):"]
entry_dist = entries["Distanz (Meilen):"]
entry_dep_delay = entries["Abflug-Verspätung (Min):"]

# Button
btn_calc = ttk.Button(main_frame, text="Berechne Ankunfts-Verspätung", command=predict_delay)
btn_calc.grid(row=9, columnspan=2, pady=20)

# Ergebnis-Label
lbl_result = ttk.Label(main_frame, text="Ergebnis erscheint hier", font=("Arial", 11, "italic"))
lbl_result.grid(row=10, columnspan=2)

root.mainloop()
