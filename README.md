# 👮 Chicago Crime Prediction AI: Analisi Predittiva e Machine Learning

# 🤝 Presentazione
Ciao, sono LAURA PERULLI | **BSc in "Statistics e Big Data"**.

In questo repository presento un progetto di **Data Science**, focalizzato sulla previsione dei crimini nella città di Chicago attraverso modelli statistici e di intelligenza artificiale.

# 🚀 Di cosa si tratta?

🔎 **ANALISI PREDITTIVA** dei crimini a Chicago:
- Realizzazione di un progetto Python incentrato sull'implementazione e l'addestramento di un modello di AI con
l'utilizzo della Regressione Logistica come algoritmo di Machine Learning.

- Analisi predittiva dei crimini a Chicago con l'utilizzo di un dataset di dati storici reali per prevedere la probabilità che si verifichi un furto (THEFT) in specifiche aree della città. 

> ### 📑 Documentazione
> Puoi visionare la Presentazione PowerPoint del Progetto:  
> 👉 [**CrimePredictionAI (PDF)**](https://github.com/user-attachments/files/24215774/CrimePredictionAI_LauraPerulli.pdf)

# 🎯 Obiettivo del progetto
1. **Prevedere la probabilità di furto**: determinare la probabilità che il crimine "THEFT" si verifichi in determinate aree di Chicago.
2. **Addestramento del modello di AI**: utilizzo del metodo di Regressione Logistica attraverso analisi e identificazione dei pattern.

# 📂 Dataset
Il dataset utilizzato è una risorsa pubblica denominata “Crimes in Chicago” proveniente dal dipartimento di polizia di Chicago e disponibile sulla piattaforma Kaggle. Raccoglie informazioni dettagliate sui crimini registrati nella città di Chicago nel periodo (2021-2025).
- Volume: Oltre 7 milioni di record;
- Campionamento: Sottoinsieme di 500.000 righe per performance migliori e campionamento di 75.000 righe per visualizzazioni geografiche ottimizzate;
- Variabili chiave: Tipo di crimine, data/ora, distretto, area comunitaria, latitudine e longitudine.

# 🛠️ Pipeline del Progetto
Il progetto segue una pipeline suddivisa in tre fasi:

**1. Analisi e pulizia del Dataset:**
   - Importazione delle librerie: (Pandas, Numpy, Matplotlib, Seaborn, Folium, Scikit-Learn).
   - Caricamento ottimizzato: analisi di solo 500.000 righe per bilanciare le performance.
   - Data Cleaning: (identificazione e rimozione di record duplicati e gestione dei valori nulli nelle colonne geografiche)
     
     ```Python
     # Rimozione record duplicati
     df.drop_duplicates(inplace=True)
     
     # Gestione dei valori nulli
     critical_subset_cols = ['Latitude', 'Longitude', 'Primary Type', 'Community Area', 'District']
     df.dropna(subset=critical_subset_cols, inplace=True)
     ```
   - Feature selection: (controllo di variabili rilevanti e rimozione di quelle irrilevanti (ID, Case Number, FBI Code, ecc...)
     
     ```Python
     # Elenco delle colonne identificate come non rilevanti per la predizione
     irrelevant = ['ID', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On', 'Location', 'Historical Wards', 'Zip Codes', 'Census Tracts', 'Wards', 'Boundaries - ZIP Codes', 'Community Areas', 'FBI Code', 'Ward', 'Beat']

     # Rimozione dinamica delle colonne se presenti nel dataframe
     columns_to_drop = [col for col in irrelevant if col in df_input.columns]
     if columns_to_drop:
     df_input.drop(columns=columns_to_drop, inplace=True)
     ```
   - Defizione della Label: (creazione di una variabile target binaria specifica per identificare i furti "THEFT" rispetto ad altre tipologie di reato, realizzazione di un GRAFICO A BARRE per visualizzare lo sbilanciamnto delle classi)
     
     ```Python
     # Creazione della variabile target: 1 se il crimine è 'THEFT', 0 altrimenti
     df['Is_Crime_Type'] = (df['Primary Type'] == crimine_da_predire).astype(int)
     
     # Calcolo conteggi e percentuali per l'analisi dello sbilanciamento
     label_counts = df['Is_Crime_Type'].value_counts()
     label_percentages = df['Is_Crime_Type'].value_counts(normalize=True) * 100
     ```
     <p align="center"> <img width="550" src="https://github.com/user-attachments/assets/ecaf26bd-4b82-4b5c-b6fe-d3031da9c926" />
   
**2. Visualizzazione dei dati e Data Analysis:**
   - Analisi geografica dei crimini a Chicago: visualizzazione delle aree più scure/rosse che evidenziano una maggiore concentrazione di crimini (mappa di calore interattiva)
     
     ```Python
     # Selezione di un sottoinsieme per ottimizzare la visualizzazione
     df_map_sample = df_map.sample(n=min(len(df_map), sample_size), random_state=42)
     
     # Creazione della mappa interattiva centrata su Chicago
     chicago_map = folium.Map(location=[df_map_sample['Latitude'].mean(), df_map_sample['Longitude'].mean()], zoom_start=11)
     
     # Preparazione e aggiunta dei dati di calore
     HeatMap(df_map_sample[['Latitude', 'Longitude']].values.tolist(), radius=15, blur=10, max_zoom=14).add_to(chicago_map)
     ```
     > ### 🗺️ Esplora la Mappa di Calore
     > Puoi consultare l'analisi geospaziale completa qui:  
     > 👉 [**Chicago Crime Heatmap (HTML)**](https://github.com/user-attachments/files/24215774/CrimePredictionAI_LauraPerulli.pdf)
     
   - Distribuzione per tipologia di Crimine evidenziando i più diffusi "THEFT", "BATTERY", "HOMECIDE":
     
     <p align="center"> <img width="550" src="https://github.com/user-attachments/assets/c30c29df-c94e-45dd-a7ed-8e800a3b0b05" />
   
   - Analisi andamento temporale dei crimini (anno, mese, giorno, ora) dei crimini e realizzazione dei GRAFICI A BARRE:
     
     ```Python
     # Estrazione delle componenti temporali (fondamentali per la predizione)
     df['Year'] = df['Date'].dt.year
     df['Month'] = df['Date'].dt.month
     df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Lunedì, 6=Domenica
     df['Hour'] = df['Date'].dt.hour
     ```
     <p align="center"> <img width="550" src="https://github.com/user-attachments/assets/ad2d4957-513e-4a82-b1df-b365e1869631" /> <img width="550" src="https://github.com/user-attachments/assets/18775b1f-1232-4452-8fd9-944ba888fd6a" /> <img width="550" src="https://github.com/user-attachments/assets/0abd1d7e-af45-4a0f-8d90-bebe0874146c" /> <img width="550" src="https://github.com/user-attachments/assets/d139b2ba-e1c4-40d1-89d1-362a8a80596f" />  </p>
     
**3. Architettura e addestramento del modello di AI:**
   - Preparazione e selezione delle feature: (gestione della stratificazione dividendo il dataset in training/set test, gestione dei valori nulli residui)
     ```Python
     # Rimozione classi rare per permettere la suddivisione stratificata
     class_counts = df['Primary Type'].value_counts()
     single_sample_classes = class_counts[class_counts == 1].index
     df_filtered = df[~df['Primary Type'].isin(single_sample_classes)].copy()

     # Imputazione dei valori mancanti nelle feature
     for col in numerical_features:
     X.loc[:, col] = X[col].fillna(X[col].mean())
     for col in categorical_features:
     X.loc[:, col] = X[col].fillna('sconosciuto').astype(str)
     ```
     
   - Addestramento del modello con Machine Learning: (Regressione Logistica configurata (class_weight='balanced'), utilizzo di Scikit-Learn per automatizzare il flusso di lavoro, uso di "StandardScaler" per variabili numeriche e "OneHotEncoder" per trasformare quelle categoriche in numeriche)
     ```Python
     # Definizione del pre-processore multimodale
     preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

     # Creazione del modello con bilanciamento delle classi
     model = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))])

     # Addestramento stratificato
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
     model.fit(X_train, y_train)
     ```

   - Valutazione e Risultati Predittivi: (classification report completo evidenziando un Accuracy di 0.32, analisi curva ROC AUC con un punteggio di 0.78 e Risk Analysis per Community Area evidenziando le probabilità predette del crimine "THEFT" visualizzando le aree più a rischio e quelle più sicure)
     ```Python
     # Valutazione accuratezza e generazione del report dettagliato
     y_pred = model.predict(X_test)
     print(f"- Accuracy: {accuracy_score(y_test, y_pred):.4f}")
     print(classification_report(y_test, y_pred, target_names=model.named_steps['classifier'].classes_))
     ```
     
     ```Python
     # Calcolo ROC AUC Score specifico per la classe 'THEFT'
      y_prob_theft = model.predict_proba(X_test)[:, theft_class_index]
      roc_auc_theft = roc_auc_score(y_test_theft_binary, y_prob_theft)

      # Generazione della curva ROC
      fpr_theft, tpr_theft, _ = roc_curve(y_test_theft_binary, y_prob_theft)
      plt.plot(fpr_theft, tpr_theft, label=f'Curva ROC THEFT (area = {roc_auc_theft:.2f})')
     ```
     
     ```Python
     # Aggregazione delle probabilità medie predette per ogni quartiere
     prob_by_comm = df_filtered.groupby('Community Area')['Predicted_Probability_THEFT'].mean().sort_values(ascending=False)

     # Visualizzazione della classifica di rischio per Community Area
     sns.barplot(x=prob_by_comm.index, y=prob_by_comm.values, palette='coolwarm')
     ```
     <p align="center"> <img width="550" src="https://github.com/user-attachments/assets/f16fef65-d95b-4ad2-9204-b4385c928a1a" /> <img width="550" src="https://github.com/user-attachments/assets/ae758f92-aa58-4b8a-8b8e-1ca1d3e65210" /> </p>
     
# ✨ Conclusioni

**🔋 Completezza del progetto 95%**
/ **📊 Comprensione dei dati 90%**
/ **📌 Precisione del modello 78%**
/ **💭 Applicabilità 85%**

Il progetto ha centrato l'obiettivo. Il valore evidenziato dell'accuratezza di 0.32 è causato dal fatto che si tratta di un modello con classi sbilanciate ma la curva ROC AUC del 0.78 è un indicatore molto più affidabile.
Attraverso un'analisi esplorativa dettagliata di un vasto dataset relativo alla criminalità di Chicago è stato sviluppato un modello predittivo basato sulla Regressione Logistica con cui è stato possibile identificare e visualizzare le tendenze spaziali e temporali della criminalità con un focus maggiore sui "THEFT". 

**PROSPETTIVE FUTURE:**
- Migliorare l'accuratezza e testare altri modelli di ML (Random Forest, KNN, Reti Neurali) e implentare un'interfaccia utente interattiva per le previsioni in tempo reale.

# 💻 Tecnologie utilizzate 
- Linguaggio di programmazione: Python
- Data Analysis: Pandas, Numpy
- Visualizzazione dati: Matplotlib, Seaborn, Folium (per mappe interattive)
- Machine Learning: Scikit-Learn

# ⚙️ Installazione
1. Scarica il dataset da [**Chicago Data Portal**](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data).
2. Installa le librerie:
  ```bash
  pip install -r requirements.txt
  ```

# **GRAZIE PER L'ATTENZIONE!** ✨
