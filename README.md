# 🔎 Analisi predittiva dei crimini a Chicago
Realizzazione di un progetto Python incentrato sull'implementazione e l'addestramento di un modello di AI con
l'utilizzo della Regressione Logistica come algoritmo di Machine Learning.

Analisi predittiva dei crimini a Chicago con l'utilizzo di un dataset di dati storici per prevedere la probabilità che si verifichi un furto (THEFT) in specifiche aree della città. 

Presentazione Power Point: [CrimePredictionAI_LauraPerulli.pdf](https://github.com/user-attachments/files/24215774/CrimePredictionAI_LauraPerulli.pdf)

# 🎯 Obiettivo del progetto:
1. Prevedere la probabilità di furto: determinare la probabilità che il crimine "THEFT" si verifichi in determinate aree di Chicago;
2. Addestramento del modello di AI: utilizzo del metodo di Regressione Logistica attraverso analisi e identificazione dei pattern.

# 📂 Dataset: 
Il dataset utilizzato è una risorsa pubblica denominata “Crimes in Chicago” proveniente dal dipartimento di polizia di Chicago e disponibile sulla piattaforma Kaggle. Raccoglie informazioni dettagliate sui crimini registrati nella città di Chicago nel periodo (2021-2025).
- Dimensione originale: Oltre 7 milioni di record;
- Dataset analizzato: Sottoinsieme di 500.000 righe per performance migliori e campionamento di 75.000 righe per visualizzazioni geografiche ottimizzate;
- Feature principali: Tipo di crimine, data/ora, distretto, area comunitaria, latitudine e longitudine.

# 🛠️ Pipeline e Codice:
Il progetto segue una pipeline suddivisa in tre fasi:
1. Analisi e pulizia del Dataset:
   - Caricamento ottimizzato: analisi di solo 500.000 righe per bilanciare le performance.
   - Data Cleaning: identificazione e rimozione di record duplicati e gestione dei valori nulli nelle colonne geografiche:
     
     ```Python
     # Rimozione record duplicati
     df.drop_duplicates(inplace=True)
     
     # Gestione dei valori nulli
     critical_subset_cols = ['Latitude', 'Longitude', 'Primary Type', 'Community Area', 'District']
     df.dropna(subset=critical_subset_cols, inplace=True)
     ```
   - Feature selection: controllo di variabili rilevanti e rimozione di quelle irrilevanti (ID, Case Number, FBI Code, ecc...):
     
     ```Python
     # Elenco delle colonne identificate come non rilevanti per la predizione
     irrelevant = [
      'ID', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate', 
      'Updated On', 'Location', 'Historical Wards', 'Zip Codes', 
      'Census Tracts', 'Wards', 'Boundaries - ZIP Codes', 
      'Community Areas', 'FBI Code', 'Ward', 'Beat'
     ]

     # Rimozione dinamica delle colonne se presenti nel dataframe
     columns_to_drop = [col for col in irrelevant if col in df_input.columns]
     if columns_to_drop:
     df_input.drop(columns=columns_to_drop, inplace=True)
     ```
   - Defizione della Label: creazione di una variabile target binaria specifica per identificare i furti "THEFT" rispetto ad altre tipologie di reato, realizzazione di un GRAFICO A BARRE per visualizzare lo sbilanciamnto delle classi:
     
     ```Python
     # Creazione della variabile target: 1 se il crimine è 'THEFT', 0 altrimenti
     df['Is_Crime_Type'] = (df['Primary Type'] == crimine_da_predire).astype(int)
     
     # Calcolo conteggi e percentuali per l'analisi dello sbilanciamento
     label_counts = df['Is_Crime_Type'].value_counts()
     label_percentages = df['Is_Crime_Type'].value_counts(normalize=True) * 100
     ```
     <img width="650" height="703" alt="DistribuzioneLabelTheft_vs_ALtri crimini" src="https://github.com/user-attachments/assets/ecaf26bd-4b82-4b5c-b6fe-d3031da9c926" />
   
3. Architettura e addestramento del modello di AI:
   - Analisi geografica dei crimini a Chicago (mappa di calore interattiva):
     
     ```Python
     # Selezione di un sottoinsieme per ottimizzare la visualizzazione
     df_map_sample = df_map.sample(n=min(len(df_map), sample_size), random_state=42)
     # Creazione della mappa interattiva centrata su Chicago
     chicago_map = folium.Map(location=[df_map_sample['Latitude'].mean(), 
                                   df_map_sample['Longitude'].mean()], 
                         zoom_start=11)
     # Preparazione e aggiunta dei dati di calore
     HeatMap(df_map_sample[['Latitude', 'Longitude']].values.tolist(),
        radius=15, blur=10, max_zoom=14).add_to(chicago_map)
     ```
     👉 Heatmap: https://lauraperulli.github.io/chicago_crime_prediction/chicago_crime_heatmap.html
     
   - Distribuzione per tipologia di Crimine evidenziando i più diffusi "THEFT", "BATTERY", "HOMECIDE":
     
     <img width="650" height="703" alt="DistribuzioneTipologiaCrimine" src="https://github.com/user-attachments/assets/c30c29df-c94e-45dd-a7ed-8e800a3b0b05" />
   
   - Analisi andamento temporale dei crimini (anno, mese, giorno, ora) dei crimini e realizzazione dei GRAFICI A BARRE:
     
     ```Python
     # Estrazione delle componenti temporali (fondamentali per la predizione)
     df['Year'] = df['Date'].dt.year
     df['Month'] = df['Date'].dt.month
     df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Lunedì, 6=Domenica
     df['Hour'] = df['Date'].dt.hour
     ```
     <img width="650" height="703" alt="AndamentoCriminiAnno" src="https://github.com/user-attachments/assets/ad2d4957-513e-4a82-b1df-b365e1869631" />
     <img width="650" height="703" alt="AndamentoCriminiMese" src="https://github.com/user-attachments/assets/18775b1f-1232-4452-8fd9-944ba888fd6a" />
     <img width="650" height="703" alt="AndamentoCriminiGiorno" src="https://github.com/user-attachments/assets/0abd1d7e-af45-4a0f-8d90-bebe0874146c" />
     <img width="650" height="703" alt="AndamentoCriminiOra" src="https://github.com/user-attachments/assets/d139b2ba-e1c4-40d1-89d1-362a8a80596f" />
     
4. Visualizzazione e valutazione dei risultati:

# 💻 Tecnologie utilizzate: 
- Linguaggio di programmazione: Python.
- Librerie: Pandas, NumPy, Scikit-Learn.
- Visualizzazione dati: Matplotlib, Seaborn, Folium (per mappe interattive).
