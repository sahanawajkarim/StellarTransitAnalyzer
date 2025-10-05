
Prepared dataset generated with notebook.
Rows: 15557
Features: obj_id, object_name, source, label, label_source, period, duration, depth, stellar_radius, stellar_mass, stellar_mag

Label mapping logic:
 - KOI: dispositions containing 'confirmed' or 'candidate' -> label 1; 'false'/'fp' -> label 0
 - TOI: TFOPWG dispositions with 'pc','kp','confirmed' -> label 1; 'fp','false' -> label 0; 'apc' ambiguous treated as 0
 - K2: similar logic to KOI

Files written:
 - preprocessed_tabular_data\prepared_dataset.csv (scaled)
 - preprocessed_tabular_data\prepared_dataset_unscaled.csv
 - preprocessed_tabular_data\train_prepared.csv
 - preprocessed_tabular_data\test_prepared.csv
 - preprocessed_tabular_data\impute_values.csv
 - preprocessed_tabular_data\scaler_params.csv

Notes:
 - Adjust label mapping functions in cell 6 if your CSVs use different wording.
 - Check column names (cell 4) if rename mappings need tweaking.
 - Deduplication kept the first occurrence per obj_id with priority KOI -> TOI -> K2.
