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
1. Analisi e Pulizia del Dataset:
   - Caricamento ottimizzato: analisi di solo 500.000 righe per bilanciare le performance;
   - Data Cleaning: identificazione e rimozione di record duplicati e gestione dei valori nulli nelle colonne geografiche.
   ```Python
   # Rimozione record duplicati
   df.drop_duplicates(inplace=True)
   
   # Gestione dei valori nulli
   critical_subset_cols = ['Latitude', 'Longitude', 'Primary Type', 'Community Area', 'District']
   df.dropna(subset=critical_subset_cols, inplace=True)
   ```
   - Feature selection: controllo di variabili rilevanti e rimozione di quelle irrilevanti (ID, Case Number, FBI Code, ecc...).
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
   - Defizione della Label: creazione di una variabile target binaria specifica per identificare i furti "THEFT" rispetto ad altre tipologie di reato, realizzazione di un grafico a barre per visualizzare lo sbilanciamnto delle classi. <img width="1366" height="703" alt="Grafico1_DistribuzioneLabelTheft_vs_ALtri crimini" src="https://github.com/user-attachments/assets/ecaf26bd-4b82-4b5c-b6fe-d3031da9c926" />
   ```Python
   # Creazione della variabile target: 1 se il crimine è 'THEFT', 0 altrimenti
   df['Is_Crime_Type'] = (df['Primary Type'] == crimine_da_predire).astype(int)
   # Calcolo conteggi e percentuali per l'analisi dello sbilanciamento
   label_counts = df['Is_Crime_Type'].value_counts()
   label_percentages = df['Is_Crime_Type'].value_counts(normalize=True) * 100
   ```
2. Architettura e Addestramento del modello di AI:
3. Visualizzazione e Valutazione dei risultati:

# 💻 Tecnologie utilizzate: 
- Linguaggio di programmazione: Python.
- Librerie: Pandas, NumPy, Scikit-Learn.
- Visualizzazione dati: Matplotlib, Seaborn, Folium (per mappe interattive).
