import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Einstellungen für schönere Plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. DATEN LADEN
# ---------------------------------------------------------
print("Lade Datensatz...")
# Ersetze 'flugdaten.csv' mit deinem Dateinamen
df = pd.read_csv('DelayedFlights.csv') 

# Wir definieren die Spalten für die detaillierten Gründe
reason_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
print("-" * 30)

# (Leeres Delay = Kein Delay?)
# ---------------------------------------------------------
print("### ANALYSE DER FEHLENDEN WERTE (NaN) IN DELAY-SPALTEN ###\n")

# Schritt A: Sind alle Reason-Spalten gleichzeitig leer?
null_mask = df[reason_cols].isnull().all(axis=1)
num_nulls = null_mask.sum()

print(f"Anzahl Flüge, wo alle Delay-Gründe LEER (NaN) sind: {num_nulls}")

# Schritt B: Wie hoch ist die Verspätung (ArrDelay) bei diesen Flügen?
# Wir schauen uns die 'ArrDelay' Statistiken für die Zeilen an, wo die Gründe leer sind
avg_delay_when_null = df.loc[null_mask, 'ArrDelay'].mean()
max_delay_when_null = df.loc[null_mask, 'ArrDelay'].max()

print(f"Durchschn. ArrDelay, wenn Gründe leer sind: {avg_delay_when_null:.2f} min")
print(f"Maximales ArrDelay, wenn Gründe leer sind: {max_delay_when_null} min")


#prüfen, ob es 'NaN'-Zeilen mit großer Verspätung (>15) gibt
unexplained_delays = df.loc[null_mask & (df['ArrDelay'] >= 15)]
print(f"Anzahl Flüge mit >15min Verspätung aber OHNE Grund: {len(unexplained_delays)}")

if len(unexplained_delays) == 0:
    print("-> ERGEBNIS: Hypothese bestätigt. Leere Gründe bedeuten 'Keine signifikante Verspätung'.")
else:
    print(f"-> ERGEBNIS: Vorsicht. Es gibt {len(unexplained_delays)} Fälle ohne Grund trotz Verspätung.")

# BEREINIGUNG FÜR WEITERE ANALYSE
# Wir füllen die NaN Werte in den Reason-Spalten mit 0, da wir nun wissen, dass sie 0 bedeuten.
df[reason_cols] = df[reason_cols].fillna(0)

print("-" * 30)

#Summe der Gründe vs. Gesamtverspätung
# ---------------------------------------------------------
# Stimmt die Summe der Gründe mit der ArrDelay überein?
df['Total_Reason_Delay'] = df[reason_cols].sum(axis=1)
df['Delay_Difference'] = df['ArrDelay'] - df['Total_Reason_Delay']

# Wir schauen uns nur Fälle an, wo es überhaupt eine Verspätung gab (>15 min)
significant_delays = df[df['ArrDelay'] >= 15]
match_rate = (significant_delays['Delay_Difference'].abs() < 2).mean() * 100 # Toleranz +/- 2 min

print(f"Bei signifikanten Verspätungen stimmt die Summe der Gründe in {match_rate:.1f}% der Fälle mit ArrDelay überein.")
print("-" * 30)


# 4. SUMMIERTE VERSPÄTUNGSGRÜNDE IN KONSOLE AUSGEBEN
# ---------------------------------------------------------
print("\n### SUMMIERTE VERSPÄTUNGSGRÜNDE (GESAMTMINUTEN) ###\n")
total_minutes_by_reason = df[reason_cols].sum().sort_values(ascending=False)

for reason, minutes in total_minutes_by_reason.items():
    print(f"{reason:20s}: {minutes:>15,.0f} Minuten")

print(f"\n{'GESAMT':20s}: {total_minutes_by_reason.sum():>15,.0f} Minuten")
print("-" * 30)


# 5. PLOTS ERSTELLEN UND SPEICHERN
# ---------------------------------------------------------
print("\nErstelle Plots...")

# A. Univariate Analyse: Verteilung der Ankunftsverspätung
plt.figure()
# Wir filtern extreme Ausreißer für die Lesbarkeit (z.B. nur bis 180 min)
sns.histplot(data=df[df['ArrDelay'].between(-30, 180)], x='ArrDelay', bins=50, kde=True)
plt.title('Verteilung der Ankunftsverspätung')
plt.xlabel('Verspätung in Minuten')
plt.axvline(0, color='red', linestyle='--') # Linie bei 0
plt.savefig('1_verteilung_verspaetung.png')
print("Plot 1 gespeichert: Verteilung")

# B. Bivariate Analyse: Verspätung nach Airline (UniqueCarrier)
plt.figure()
# Sortieren nach Median-Verspätung
order = df.groupby('UniqueCarrier')['ArrDelay'].median().sort_values().index
sns.boxplot(data=df, x='UniqueCarrier', y='ArrDelay', order=order, showfliers=False) 
# showfliers=False blendet die extremen Punkte aus, damit man die Boxen besser sieht
plt.title('Verspätungsverteilung pro Airline')
plt.savefig('2_verspaetung_nach_airline.png')
print("Plot 2 gespeichert: Airline Vergleich")

# C. Analyse der Gründe: Was verursacht die meiste Zeit?
# Wir summieren die Minuten pro Grund auf
plt.figure()
total_minutes_by_reason.plot(kind='bar', color='salmon')
plt.title('Gesamtminuten Verspätung nach Ursache')
plt.ylabel('Minuten (Summe)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('3_verspaetungsgruende.png')
print("Plot 3 gespeichert: Gründe")

# D. Zeitreihe: Durchschnittliche Verspätung pro Wochentag
# DayOfWeek: 1=Montag, 7=Sonntag
plt.figure()
avg_delay_day = df.groupby('DayOfWeek')['ArrDelay'].mean()
avg_delay_day.plot(kind='line', marker='o', color='green')
plt.title('Durchschnittliche Verspätung nach Wochentag')
plt.ylabel('Durchschnittliche Verspätung (min)')
plt.xticks(range(1,8), ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So'])
plt.grid(True)
plt.savefig('4_wochentagstrend.png')
print("Plot 4 gespeichert: Wochentag")

print("\nFertig! Alle Bilder wurden im Ordner gespeichert.")
