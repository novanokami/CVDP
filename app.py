from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HeartMLP(nn.Module):
    def __init__(self, input_size: int, class_count: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, class_count),
        )

    def forward(self, features):
        return self.layers(features)


def make_preprocessor(numeric_columns, categorical_columns):
    transforms = []
    if numeric_columns:
        transforms.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric_columns)))
    if categorical_columns:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_columns)))
    return ColumnTransformer(transforms, sparse_threshold=0.0)


def add_derived_features(frame):
    """Add simple measurements that may help the network model useful relationships."""
    result = frame.copy()
    columns = {column.strip().lower(): column for column in result.columns}
    if "height" in columns and "weight" in columns:
        height_m = pd.to_numeric(result[columns["height"]], errors="coerce") / 100.0
        weight_kg = pd.to_numeric(result[columns["weight"]], errors="coerce")
        result["bmi_from_height_weight"] = weight_kg / height_m.pow(2).replace(0, np.nan)
    systolic = next((columns[name] for name in ("ap_hi", "trtbps") if name in columns), None)
    diastolic = columns.get("ap_lo")
    if systolic and diastolic:
        high = pd.to_numeric(result[systolic], errors="coerce")
        low = pd.to_numeric(result[diastolic], errors="coerce")
        result["pulse_pressure_from_bp"] = high - low
        result["mean_arterial_pressure_from_bp"] = (high + 2 * low) / 3
    return result


@st.cache_resource(show_spinner="Training the neural network on the available device...")
def train_model(X_train, y_train, numeric_columns, categorical_columns):
    fit_X, validation_X, fit_y, validation_y = train_test_split(
        X_train, y_train, test_size=0.15, random_state=43, stratify=y_train
    )
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    train_features = preprocessor.fit_transform(fit_X).astype(np.float32)
    validation_features = preprocessor.transform(validation_X).astype(np.float32)
    label_encoder = LabelEncoder().fit(y_train)
    train_labels = label_encoder.transform(fit_y).astype(np.int64)
    validation_labels = label_encoder.transform(validation_y).astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    network = HeartMLP(train_features.shape[1], len(label_encoder.classes_)).to(device)
    x_tensor = torch.from_numpy(train_features).to(device)
    y_tensor = torch.from_numpy(train_labels).to(device)
    x_validation = torch.from_numpy(validation_features).to(device)
    y_validation = torch.from_numpy(validation_labels).to(device)
    counts = torch.bincount(y_tensor, minlength=len(label_encoder.classes_)).float()
    class_weights = (len(y_tensor) / (len(counts) * counts)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    batch_size = min(4096, max(64, len(y_tensor)))
    max_epochs = 60
    patience = 10
    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        network.train()
        order = torch.randperm(len(y_tensor), device=device)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(network(x_tensor[indices]), y_tensor[indices])
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            validation_loss = loss_fn(network(x_validation), y_validation).item()
        scheduler.step(validation_loss)
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = {name: value.detach().clone() for name, value in network.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_weights is not None:
        network.load_state_dict(best_weights)
    network.eval()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (CUDA GPU not detected)"
    return preprocessor, label_encoder, network, device, device_name, best_epoch


@st.cache_resource(show_spinner="Training comparison models on the same training split...")
def train_comparison_models(X_train, y_train, numeric_columns, categorical_columns):
    configurations = {
        "Logistic regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.07, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, random_state=42,
        ),
    }
    fitted = {}
    for name, classifier in configurations.items():
        estimator = Pipeline([
            ("preprocessor", make_preprocessor(numeric_columns, categorical_columns)),
            ("classifier", classifier),
        ])
        fitted[name] = estimator.fit(X_train, y_train)
    return fitted


def model_outputs(bundle, frame):
    preprocessor, label_encoder, network, device, _, _ = bundle
    matrix = preprocessor.transform(add_derived_features(frame)).astype(np.float32)
    with torch.inference_mode():
        logits = network(torch.from_numpy(matrix).to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = label_encoder.inverse_transform(probabilities.argmax(axis=1))
    return predictions, probabilities


@st.cache_resource
def build_rag_index(chunks):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve_project_context(question, chunks, vectorizer, document_matrix):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, document_matrix).ravel()
    best = np.argsort(scores)[-3:][::-1]
    context = "\n\n".join(chunks[index] for index in best if scores[index] > 0)
    if not context:
        context = "The project knowledge base has no directly matching passage. Be transparent about this."
    return context


def offline_project_answer(question, chunks, vectorizer, document_matrix):
    context = retrieve_project_context(question, chunks, vectorizer, document_matrix)
    if context.startswith("The project knowledge base has no directly matching passage"):
        return "I couldn't find a matching passage in the local project notes. Try asking about the app, dataset columns, training, or model metrics. I can answer only from the notes included with this project."
    return "Here is the closest information from the project's local notes:\n\n" + context


HEALTH_TOPIC_GUIDES = {
    "High blood pressure": {
        "about": "Blood pressure that stays high can strain and damage blood vessels over time. It is associated with higher risk of heart disease and stroke, and often has no obvious symptoms.",
        "clinician": "A clinician may repeat properly taken blood-pressure readings or recommend home or 24-hour monitoring to check whether it is persistently high. They will interpret readings in context.",
        "general": "Keep a record of readings and discuss them with a healthcare professional. Ask what lifestyle changes or treatment are appropriate; do not change medication based on this app.",
        "source": "[CDC: High blood pressure](https://www.cdc.gov/high-blood-pressure/index.html)",
    },
    "High cholesterol": {
        "about": "High LDL cholesterol can contribute to plaque buildup in artery walls. This can narrow blood vessels and is associated with heart disease and stroke. High cholesterol usually has no symptoms.",
        "clinician": "A clinician may discuss a lipid profile blood test, which measures LDL, HDL, total cholesterol, and triglycerides, and interpret it alongside other factors.",
        "general": "Ask a healthcare professional how often testing is appropriate and what the results mean for you. Treatment choices are individual; do not start or stop medication from an app result.",
        "source": "[CDC: Testing for cholesterol](https://www.cdc.gov/cholesterol/testing/index.html)",
    },
    "High blood sugar or diabetes": {
        "about": "Over time, high blood sugar can damage blood vessels and nerves that control the heart. Diabetes often occurs with other cardiovascular risk factors, such as high blood pressure or cholesterol.",
        "clinician": "A clinician may review glucose measurements and consider blood tests such as A1C or fasting glucose, based on your situation and history.",
        "general": "Discuss test results and a follow-up plan with a healthcare professional. Ask about heart-health checks and support for managing blood sugar; this assistant cannot assess your personal risk.",
        "source": "[CDC: Diabetes and your heart](https://www.cdc.gov/diabetes/diabetes-complications/diabetes-and-your-heart.html) · [CDC: Diabetes testing](https://www.cdc.gov/diabetes/diabetes-testing/index.html)",
    },
    "Smoking or tobacco use": {
        "about": "Smoking harms blood vessels and increases the risk of cardiovascular disease. Quitting can improve health, and support is available.",
        "clinician": "A healthcare professional can review cardiovascular risk factors and discuss counseling, quit-support services, and whether cessation medicines are suitable.",
        "general": "If you want to quit, ask a clinician or local tobacco-cessation service about a plan and support. Avoid relying on this project to choose medicines.",
        "source": "[CDC: Smoking and cardiovascular disease](https://www.cdc.gov/tobacco/about/cigarettes-and-cardiovascular-disease.html)",
    },
    "Several risk factors": {
        "about": "Factors such as high blood pressure, high cholesterol, diabetes, smoking, and low physical activity can combine. An app cannot calculate a reliable personal future-event risk from this selection alone.",
        "clinician": "A clinician can review your history, measurements, laboratory results, and family history together and decide whether additional checks are appropriate.",
        "general": "Bring existing readings and test results to an appointment. Ask which factors matter most for you and what follow-up is appropriate. Do not change medication based on this app.",
        "source": "[CDC: Diabetes and your heart](https://www.cdc.gov/diabetes/diabetes-complications/diabetes-and-your-heart.html) · [CDC: High blood pressure](https://www.cdc.gov/high-blood-pressure/index.html)",
    },
    "I have urgent symptoms right now": {
        "about": "This app cannot assess symptoms. New or severe chest pain or pressure, trouble breathing, fainting, or sudden face/arm weakness or speech trouble may need emergency care.",
        "clinician": "Do not wait for an app response or an online conversation to decide what is happening.",
        "general": "If symptoms are severe, sudden, or concerning, contact your local emergency service now. Do not drive yourself if you may be having an emergency.",
        "source": "[American Heart Association: Heart attack warning signs](https://www.heart.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack) · [American Heart Association: Heart attack and stroke symptoms](https://www.heart.org/en/about-us/heart-attack-and-stroke-symptoms)",
    },
}


def format_health_topic_guide(topic):
    guide = HEALTH_TOPIC_GUIDES[topic]
    return (
        f"### {topic}\n\n"
        f"**How it may relate to heart health:** {guide['about']}\n\n"
        f"**What a clinician may check or discuss:** {guide['clinician']}\n\n"
        f"**General next step:** {guide['general']}\n\n"
        f"**Trusted information:** {guide['source']}\n\n"
        "*Educational information only; this is not a diagnosis or personal risk estimate.*"
    )


st.set_page_config(page_title="Cardiovascular Disease Prediction", page_icon="❤️", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(120deg, rgba(23, 72, 95, .42), rgba(24, 38, 63, .72));
        border: 1px solid rgba(112, 190, 202, .23);}
    .hero h1 {margin: 0 0 .35rem 0;}
    .hero p {margin: 0; color: #cbd5e1; font-size: 1.04rem;}
    [data-testid="stMetric"] {background: rgba(255,255,255,.035); padding: .8rem 1rem;
        border: 1px solid rgba(255,255,255,.08); border-radius: 12px;}
</style>
<div class="hero"><h1>❤️ Cardiovascular Health Learning Lab</h1>
<p>Explore a locally trained neural network, review its evaluation, and learn what dataset indicators can—and cannot—tell you.</p></div>
""", unsafe_allow_html=True)
st.warning("Educational demonstration only. This is not a medical device and must not be used to diagnose, treat, or rule out disease. Consult a qualified healthcare professional.")

data_dir = Path(__file__).parent / "data"
available_datasets = {
    "Cardio training dataset (70,000 rows)": data_dir / "cardio_train.csv",
    "Heart practice dataset (289 rows)": data_dir / "heart_practice.csv",
}
uploaded = st.file_uploader("Upload another labeled CSV (optional)", type=["csv"])
with st.sidebar:
    dataset_choice = st.selectbox("Training dataset", list(available_datasets))
selected_data = available_datasets[dataset_choice]
if uploaded is None and not selected_data.exists():
    st.info("Upload a labeled dataset with feature columns and an outcome column such as cardio, output, or target.")
    st.stop()
try:
    data = pd.read_csv(uploaded if uploaded is not None else selected_data, sep=None, engine="python")
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()
id_cols = [column for column in data.columns if column.strip().lower() in {"id", "index", "unnamed: 0"}]
data = data.drop(columns=id_cols)
if data.empty or len(data.columns) < 2:
    st.error("The CSV must contain at least one feature and one outcome column.")
    st.stop()
with st.sidebar:
    st.header("Configure model")
    known_targets = [column for column in ("cardio", "output", "target") if column in data.columns]
    if known_targets:
        target = known_targets[0]
        st.info(f"Prediction target is fixed to `{target}` for this dataset.")
    else:
        target = st.selectbox("Outcome / target column", data.columns, index=0)
    test_size = st.slider("Test set size", 0.15, 0.40, 0.20, 0.05)

clean = data.dropna(subset=[target]).copy()
y = clean[target]
if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("The outcome needs at least two classes, with at least two rows in each class.")
    st.stop()
feature_names = [column for column in clean.columns if column != target]
age_col = next((column for column in feature_names if column.strip().lower() == "age"), None)
if age_col and pd.to_numeric(clean[age_col], errors="coerce").median() > 120:
    clean[age_col] = (pd.to_numeric(clean[age_col], errors="coerce") / 365.25).round(1)
    st.info("Age values were converted from days to years based on the dataset's age range.")
X = add_derived_features(clean[feature_names])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
numeric = X_train.select_dtypes(include="number").columns.tolist()
categorical = [column for column in feature_names if column not in numeric]
try:
    bundle = train_model(X_train, y_train, tuple(numeric), tuple(categorical))
except Exception as exc:
    st.error(f"Model training failed. Details: {exc}")
    st.stop()
preprocessor, label_encoder, network, device, device_name, best_epoch = bundle
test_predictions, test_probabilities = model_outputs(bundle, X_test)
st.subheader("Model check")
st.caption(f"PyTorch neural network · {len(clean):,} labeled rows · training device: {device_name} · best validation epoch: {best_epoch}")
metric_cols = st.columns(3)
metric_cols[0].metric("Test accuracy", f"{accuracy_score(y_test, test_predictions):.1%}")
if len(label_encoder.classes_) == 2:
    try:
        encoded_test = label_encoder.transform(y_test)
        metric_cols[1].metric("ROC AUC", f"{roc_auc_score(encoded_test, test_probabilities[:, 1]):.3f}")
    except ValueError:
        metric_cols[1].metric("ROC AUC", "Unavailable")
else:
    metric_cols[1].metric("Outcome classes", str(len(label_encoder.classes_)))
metric_cols[2].metric("Test records", f"{len(y_test):,}")
st.caption("These are holdout-set metrics, not a measure of clinical usefulness. Accuracy can mislead when classes are imbalanced; review per-class metrics and the confusion matrix.")
left, right = st.columns(2)
with left:
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y_test, test_predictions, output_dict=True, zero_division=0)).transpose().round(3), use_container_width=True)
with right:
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions, labels=label_encoder.classes_), display_labels=label_encoder.classes_).plot(ax=ax, colorbar=False)
    ax.set_title("Test set confusion matrix")
    st.pyplot(fig, clear_figure=True)

st.subheader("Compare models on the same held-out records")
st.caption("All models use the same training/test split. The test records are kept separate from training; use the table to compare strengths, not to claim clinical performance.")
try:
    comparison_models = train_comparison_models(X_train, y_train, tuple(numeric), tuple(categorical))
    comparison_predictions = {"PyTorch neural network": test_predictions}
    comparison_probabilities = {"PyTorch neural network": test_probabilities}
    for model_name, estimator in comparison_models.items():
        comparison_predictions[model_name] = estimator.predict(X_test)
        comparison_probabilities[model_name] = estimator.predict_proba(X_test)

    y_encoded = label_encoder.transform(y_test)
    positive_label = label_encoder.classes_[1] if len(label_encoder.classes_) == 2 else None
    comparison_rows = []
    for model_name, model_prediction in comparison_predictions.items():
        encoded_prediction = label_encoder.transform(model_prediction)
        average = "binary" if len(label_encoder.classes_) == 2 else "weighted"
        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, model_prediction),
            "Precision": precision_score(y_test, model_prediction, average=average, pos_label=positive_label if positive_label is not None else 1, zero_division=0),
            "Recall": recall_score(y_test, model_prediction, average=average, pos_label=positive_label if positive_label is not None else 1, zero_division=0),
            "F1 score": f1_score(y_test, model_prediction, average=average, pos_label=positive_label if positive_label is not None else 1, zero_division=0),
        }
        try:
            probabilities = comparison_probabilities[model_name]
            if len(label_encoder.classes_) == 2:
                row["ROC AUC"] = roc_auc_score(y_encoded, probabilities[:, 1])
            else:
                row["ROC AUC"] = roc_auc_score(y_encoded, probabilities, multi_class="ovr", average="weighted")
        except ValueError:
            row["ROC AUC"] = np.nan
        comparison_rows.append(row)

    comparison_table = pd.DataFrame(comparison_rows).sort_values("Accuracy", ascending=False)
    st.dataframe(comparison_table.set_index("Model").style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(comparison_table.set_index("Model")[["Accuracy", "F1 score", "ROC AUC"]])
    st.caption("Accuracy is the share classified correctly. Recall shows how many true positive-class records were found; precision shows how many positive predictions were correct. Choose metrics that match the project question.")
    with st.expander("Explore which fields influence the top model's predictions"):
        st.write("Permutation importance measures how much the selected model's test accuracy changes when one column is shuffled. It describes patterns in this dataset; it does not show that a feature causes disease.")
        importance_model_name = st.selectbox("Choose a model to explain", list(comparison_models), key="importance_model")
        if st.button("Calculate feature importance", key="calculate_permutation_importance"):
            importance_result = permutation_importance(
                comparison_models[importance_model_name], X_test, y_test,
                scoring="accuracy", n_repeats=3, random_state=42,
                n_jobs=-1, max_samples=min(3000, len(X_test)),
            )
            importance_table = pd.DataFrame({
                "Feature": X_test.columns,
                "Accuracy decrease when shuffled": importance_result.importances_mean,
            }).sort_values("Accuracy decrease when shuffled", ascending=False).head(10)
            st.dataframe(importance_table.set_index("Feature").style.format("{:.4f}"), use_container_width=True)
except Exception as exc:
    st.warning(f"Could not train all comparison models: {exc}")

st.divider()
st.header("1 · Explore a dataset prediction")
st.write(f"Enter one row using the units and category codes documented by your dataset. The model predicts the dataset's `{target}` label; it does not predict a heart attack or diagnose diabetes or high cholesterol.")
values = {}
with st.container(border=True):
    st.markdown("**Measurements**")
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, name in enumerate(feature_names):
            series = X_train[name]
            with columns[index % 2]:
                if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
                    median = float(series.median()) if series.notna().any() else 0.0
                    values[name] = st.number_input(name, value=median, help=f"Training range: {series.min()} to {series.max()}")
                elif pd.api.types.is_numeric_dtype(series):
                    options = sorted(series.dropna().unique().tolist())
                    values[name] = st.selectbox(name, options or [0])
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist()) or [""]
                    values[name] = st.selectbox(name, options)
        submitted = st.form_submit_button("Run educational prediction", type="primary")
if submitted:
    st.session_state["last_prediction_values"] = values.copy()
if "last_prediction_values" in st.session_state:
    prediction_values = st.session_state["last_prediction_values"]
    row = pd.DataFrame([prediction_values], columns=feature_names)
    prediction, probability = model_outputs(bundle, row)
    st.divider()
    st.header("2 · Prediction result and heart-health learning")
    st.caption("This is the model's predicted dataset class—not a heart-attack probability or a personal clinical risk score.")
    result_cols = st.columns([1, 2])
    with result_cols[0]:
        with st.container(border=True):
            st.markdown("**Dataset model result**")
            st.metric("Predicted class", str(prediction[0]))
            if len(label_encoder.classes_) == 2:
                st.write(f"Model probability for class {label_encoder.classes_[1]}: **{probability[0, 1]:.1%}**")
                st.caption("Model confidence can be wrong and is not an individual's disease probability.")
    with result_cols[1]:
        with st.container(border=True):
            st.markdown("**How to read the factor cards**")
            st.write("Cards show values stored in the selected CSV. Categorical codes are not lab readings, and numeric values are not clinically interpreted here. Check the source data dictionary; these cards do not confirm a diagnosis.")
    cholesterol_column = next((name for name in feature_names if name.strip().lower() in {"cholesterol", "chol"}), None)
    glucose_column = next((name for name in feature_names if name.strip().lower() in {"gluc", "glucose"}), None)
    fasting_sugar_column = next((name for name in feature_names if name.strip().lower() == "fbs"), None)

    def numeric_code(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    st.subheader("Factor context, precautions, and next steps")
    st.info("These cards explain general health connections and safer next steps. The selected CSV does not track future heart attacks, strokes, or heart failure, so this app cannot predict whether a person will develop them.")
    st.caption("Category mappings vary by dataset. Do not use these cards to diagnose, treat, or calculate an individual’s risk.")
    factor_columns = st.columns(3)
    if cholesterol_column:
        code = numeric_code(prediction_values[cholesterol_column])
        with factor_columns[0].container(border=True):
            st.markdown("#### 🧬 Cholesterol")
            if clean[cholesterol_column].nunique() > 10:
                st.metric("Recorded dataset value", str(prediction_values[cholesterol_column]))
                st.write("This is a numeric dataset field. Confirm its unit and meaning in the source documentation; the app does not assess whether it is medically high.")
                st.markdown("**Possible heart connection:** Persistently high LDL cholesterol can contribute to plaque in arteries, which may narrow blood flow to the heart or brain.")
                st.markdown("**Possible test:** A lipid panel measures total cholesterol, LDL, HDL, and triglycerides. Ask whether you need to fast.")
                st.markdown("**Advice a clinician may discuss:** Your overall heart-health picture and whether lifestyle changes or treatment are appropriate after reviewing actual results.")
            elif code in (2, 3):
                level = "above normal" if code == 2 else "well above normal"
                st.warning(f"Dataset category: {level}")
                st.write("This category is not an LDL/HDL lab result. If a real test confirms unhealthy cholesterol over time, LDL can contribute to plaque that narrows arteries and is associated with heart disease and stroke.")
                st.markdown("**Possible test:** A lipid panel measures total cholesterol, LDL, HDL, and triglycerides; the clinician may ask about preparation or fasting.")
                st.markdown("**Advice a clinician may discuss:** How your actual results and other factors affect next steps. General habits may include balanced eating, suitable activity, and avoiding tobacco.")
            elif code == 1:
                st.info("Dataset category: 1. It does not replace a lipid-panel result or rule out risk. Ask a clinician how often testing is appropriate for you.")
            else:
                st.info("Check the source data dictionary before interpreting this category.")
            if code not in (2, 3) and clean[cholesterol_column].nunique() <= 10:
                st.markdown("**Possible clinician follow-up:** They may consider a lipid panel and review your overall health, depending on your history.")
            st.caption("Do not start, stop, or change cholesterol medicine based on this app.")

    if glucose_column:
        code = numeric_code(prediction_values[glucose_column])
        with factor_columns[1].container(border=True):
            st.markdown("#### 🍬 Blood glucose")
            if code in (2, 3):
                level = "above normal" if code == 2 else "well above normal"
                st.warning(f"Dataset category: {level}")
                st.write("This category does not diagnose diabetes. Persistently high blood sugar in diabetes can damage blood vessels and nerves; diabetes is associated with higher heart-disease and stroke risk.")
                st.markdown("**Possible tests:** An A1C test or laboratory blood-glucose test; the clinician chooses based on your history and may confirm an abnormal result.")
                st.markdown("**Advice a clinician may discuss:** Follow-up and a care plan based on test results. If already diagnosed, follow your own healthcare team’s plan.")
            elif code == 1:
                st.info("Dataset category: 1. This is not a diabetes test and does not rule out disease. Ask a clinician if you need screening.")
            else:
                st.info("Check the source data dictionary before interpreting this category.")
            if code not in (2, 3):
                st.markdown("**Possible clinician follow-up:** They may consider A1C or blood-glucose testing depending on your health history.")
            st.caption("Do not start, stop, or change diabetes medicines or treatment based on this app.")
    elif fasting_sugar_column:
        with factor_columns[1].container(border=True):
            st.markdown("#### 🍬 Fasting blood sugar field")
            st.metric("Recorded dataset value", str(prediction_values[fasting_sugar_column]))
            st.write("This dataset field is not a diabetes diagnosis. Its exact coding and threshold depend on the source documentation.")
            st.markdown("**Possible heart connection:** Diabetes and persistently high blood sugar can be associated with blood-vessel damage and greater heart-disease risk, but this coded field alone cannot establish either.")
            st.markdown("**Possible tests:** A1C or laboratory blood-glucose testing, chosen by a healthcare professional based on your situation.")
            st.markdown("**Advice a clinician may discuss:** Whether another test or follow-up is appropriate. This dataset flag is not enough to diagnose diabetes.")
            st.caption("Precaution: do not infer diabetes from this model input.")

    systolic_column = next((name for name in feature_names if name.strip().lower() in {"ap_hi", "trtbps"}), None)
    diastolic_column = next((name for name in feature_names if name.strip().lower() == "ap_lo"), None)
    if systolic_column:
        with factor_columns[2].container(border=True):
            st.markdown("#### 🩺 Blood pressure")
            bp_value = f"{prediction_values[systolic_column]} / {prediction_values[diastolic_column]}" if diastolic_column else str(prediction_values[systolic_column])
            st.metric("Recorded dataset value", bp_value)
            st.write("Check the source documentation for units and whether this field is a single reading or a pair. One entry here cannot diagnose hypertension or determine future outcomes.")
            st.markdown("**Possible heart connection:** Blood pressure that stays high can damage artery walls and is associated with heart disease and stroke.")
            st.markdown("**Possible checks:** Repeated, properly taken readings or home/24-hour monitoring to confirm the pattern.")
            st.markdown("**Advice a clinician may discuss:** How to track readings and whether lifestyle changes or treatment are appropriate after reviewing them.")

    next_steps = [
        "For personal advice, bring actual blood-pressure readings and laboratory results to a healthcare professional. They can interpret glucose, cholesterol, and other factors together.",
        "General prevention habits can include a balanced eating pattern, regular physical activity suitable for you, and avoiding tobacco. Ask your healthcare team what changes are appropriate for your health.",
        "If you smoke, ask a healthcare professional about quit-support options. Many adults are encouraged to aim for about 150 minutes of moderate activity per week, but choose activity with your care team if you have health concerns.",
        "Do not start, stop, or change medication based on this app. Discuss treatment options with a qualified healthcare professional.",
    ]
    with st.container(border=True):
        st.markdown("### Practical heart-health steps")
        st.write("These are general prevention ideas, not a personalized treatment plan:")
        st.markdown("\n".join(f"- {item}" for item in next_steps))
    st.markdown("Sources: [CDC: Diabetes and your heart](https://www.cdc.gov/diabetes/diabetes-complications/diabetes-and-your-heart.html) · [CDC: Diabetes tests](https://www.cdc.gov/diabetes/diabetes-testing/index.html) · [CDC: Cholesterol testing](https://www.cdc.gov/cholesterol/testing/index.html) · [AHA: Blood pressure measurement](https://professional.heart.org/en/science-news/measurement-of-blood-pressure-in-humans)")

st.divider()
st.header("3 · Ask the local project assistant")
st.write("Choose a common heart-health concern for a basic explanation, or ask a question about the app, dataset, or model.")
st.info("This assistant works offline. Topic guidance is prepared educational information; project questions use TF-IDF search through local notes. It does not diagnose or estimate personal risk. Avoid entering names or detailed personal health information.")

with st.form("health_topic_form"):
    selected_health_topic = st.selectbox(
        "What would you like to understand?",
        list(HEALTH_TOPIC_GUIDES),
        help="Choose a general topic. This does not tell the app that you have the condition.",
    )
    show_topic_guide = st.form_submit_button("Show basic information")

if show_topic_guide:
    if selected_health_topic == "I have urgent symptoms right now":
        st.error(format_health_topic_guide(selected_health_topic))
    else:
        st.markdown(format_health_topic_guide(selected_health_topic))
    st.caption("The information above is general and may not fit an individual. For personal concerns, contact a qualified healthcare professional.")

st.markdown("#### Ask about the project")

readme_path = Path(__file__).parent / "README.md"
knowledge_chunks = []
if readme_path.exists():
    knowledge_chunks.extend(chunk.strip() for chunk in readme_path.read_text(encoding="utf-8").split("\n\n") if len(chunk.strip()) > 30)
target_counts = clean[target].value_counts().to_dict()
knowledge_chunks.append(
    f"Dataset summary: {len(clean)} rows. Outcome column: {target}. Outcome class counts: {target_counts}. "
    f"Input columns: {', '.join(feature_names)}. Age is converted from days to years when the values indicate days. "
    "The app trains a PyTorch multilayer neural network using a stratified train/test split. "
    "Feature preprocessing imputes missing values, scales numeric columns, and one-hot encodes text categories."
)
knowledge_chunks.append(
    "Dataset field meanings and category codes depend on the dataset documentation. Do not infer a person's diagnosis "
    "or clinical risk from this educational model. The model is not clinically validated."
)
vectorizer, document_matrix = build_rag_index(tuple(knowledge_chunks))
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
for message in st.session_state.assistant_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input("Ask about the app, dataset, or model")
if question:
    user_message = {"role": "user", "content": question}
    with st.chat_message("user"):
        st.markdown(question)
    answer = offline_project_answer(question, knowledge_chunks, vectorizer, document_matrix)
    st.session_state.assistant_messages.extend([user_message, {"role": "assistant", "content": answer}])
    with st.chat_message("assistant"):
        st.markdown(answer)

st.caption("Use only data you are authorized to process. Do not upload identifiable patient information to an environment that is not approved for that use.")
