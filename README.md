# Cardiovascular Disease Prediction Using ML

A Python and Streamlit educational project for comparing several classifiers on a labeled CSV, reviewing holdout-set metrics and feature importance, and exploring a sample prediction.

## Run in Visual Studio Code

Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in your browser with a dataset selector. It includes `data/cardio_train.csv` (70,000 rows, semicolon-separated) and `data/heart_practice.csv` (289 rows, comma-separated, with an `output` target); you can also upload another labeled CSV. The heart-practice file you supplied has 13 feature columns and 1 target column. Its small sample size can make holdout metrics unstable and neural networks prone to overfitting, so do not assume that a larger model or a good score means better real-world predictions. The cardio dataset's `age` column is stored in days, so the app converts it to years for training and entry. Known target columns (`cardio`, `output`, or `target`) are selected automatically so a feature column is not accidentally used as the outcome.

## Dataset

The project owner reports that the dataset is freely available and that permission to use it has been obtained, with proof. Keep that proof and the dataset's source citation with the project when sharing it. Read the dataset documentation and confirm target meanings, measurement units, and category codes before interpreting results. Common row identifiers named `id`, `index`, or `Unnamed: 0` are dropped. The model imputes missing feature values, scales numeric columns, one-hot encodes text categories, and trains a neural network with a stratified holdout split.

The app compares a PyTorch neural network, logistic regression, random forest, and histogram gradient boosting on the same stratified train/test split. It reports accuracy, precision, recall, F1, ROC AUC, a classification report, and a confusion matrix. An optional permutation-importance view shows how shuffling an input changes a selected model's accuracy; this is a dataset pattern, not a cause of disease. Metrics can be misleading when data is imbalanced or biased and should not be read as clinical performance. For the cardio dataset, the app derives BMI from height/weight and pulse pressure and mean arterial pressure from systolic/diastolic pressure. Neural-network training uses an internal validation split, a learning-rate scheduler, and early stopping to limit overfitting. It uses CUDA when a compatible GPU-enabled PyTorch installation is available and otherwise falls back to CPU.

After a prediction, the app shows educational context cards for matching cholesterol, glucose, fasting-sugar, and blood-pressure fields. Cards explain possible heart-health connections, tests a clinician may consider, and follow-up advice a clinician may discuss, with links to public-health sources. These tests are examples, not instructions for a particular person. The app does not predict future heart attacks, strokes, or heart failure: the included datasets do not contain longitudinal future-event labels. Categorical codes and numeric measurements depend on the selected dataset's data dictionary and are not diagnoses. The guidance is informational, not a personalized treatment plan; medication decisions belong with a qualified healthcare professional.

## Local GPU setup and offline project assistant

The assistant works offline and makes no cloud or local model API calls. Its inquiry box uses TF-IDF retrieval to find relevant passages in the README and dataset summary. It also has a choice-based health-topic guide for general information about high blood pressure, cholesterol, blood sugar/diabetes, smoking, multiple risk factors, and urgent symptoms. The topic guide explains possible heart-health connections, what a clinician may discuss, and general next steps; it does not diagnose or estimate individual risk. The project-question box returns matching notes instead of generating free-form answers. No API key, Ollama, or model download is needed.

To use the GPU for prediction training, install a CUDA-enabled PyTorch build if you have a compatible NVIDIA GPU. Use the official [PyTorch install selector](https://pytorch.org/get-started/locally/) with Windows, Pip, and the CUDA option that matches your system. Then check whether PyTorch detects it:

   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
   ```

Start the app with `.\.venv\Scripts\python.exe -m streamlit run app.py`. The app shows which device trained the predictor.

The assistant is intentionally restricted to project help and must not be used for diagnosis, treatment, or personal risk advice. Never enter personal health information or symptoms.

## Important

This is an educational example, not a medical device. It has not been clinically validated and must not be used to diagnose, treat, or rule out disease. Predictions are not a substitute for a clinician's assessment. Do not upload identifiable patient data unless the environment is approved for that use.

