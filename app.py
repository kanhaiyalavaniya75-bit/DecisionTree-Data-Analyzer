import json
import io
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL = {"pipeline": None, "features": [], "target": None, "label_encoder": None,
         "mapping": {}, "tree_feature_names": [], "classes": []}


def read_dataset(file_obj):
    if file_obj is None:
        raise ValueError("Please upload a CSV dataset first.")
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    df = pd.read_csv(path)
    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV must contain at least two columns and some rows.")
    return df


def inspect_dataset(file_obj):
    try:
        df = read_dataset(file_obj)
        info = f"Rows: {len(df)} | Columns: {len(df.columns)}\n\nColumns:\n" + "\n".join(
            f"- {c}{' (target candidate)' if df[c].nunique(dropna=True) == 2 else ''}" for c in df.columns
        )
        return df.head(10), gr.update(choices=list(df.columns), value=list(df.columns)[-1]), info
    except Exception as e:
        return None, gr.update(choices=[], value=None), f"Error: {e}"


def train_model(file_obj, target_col, test_size, max_depth, min_samples_leaf):
    try:
        df = read_dataset(file_obj)
        if not target_col or target_col not in df.columns:
            raise ValueError("Select a valid target column.")
        df = df.dropna(subset=[target_col]).copy()
        y_raw = df[target_col].astype(str).str.strip()
        classes = list(pd.unique(y_raw))
        if len(classes) != 2:
            raise ValueError(f"Target column must have exactly 2 classes for Yes/No output. Found {len(classes)}: {classes}")

        # Preserve explicit Yes/No labels; otherwise map the two classes consistently to No/Yes.
        lower_to_original = {str(v).strip().lower(): v for v in classes}
        if "yes" in lower_to_original and "no" in lower_to_original:
            positive_label = lower_to_original["yes"]
            negative_label = lower_to_original["no"]
        else:
            sorted_classes = sorted(classes, key=lambda x: str(x).lower())
            negative_label, positive_label = sorted_classes[0], sorted_classes[1]

        y = y_raw.map({str(negative_label): 0, str(positive_label): 1}).astype(int)
        X = df.drop(columns=[target_col]).copy()
        if X.shape[1] == 0:
            raise ValueError("Dataset needs at least one feature column besides the target.")

        # Remove columns that contain no usable values.
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0:
            raise ValueError("No usable feature columns were found.")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        transformers = []
        if numeric_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        clf = DecisionTreeClassifier(max_depth=int(max_depth) if max_depth else None,
                                     min_samples_leaf=int(min_samples_leaf), random_state=42)
        pipeline = Pipeline([("preprocess", preprocessor), ("tree", clf)])

        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=42, stratify=stratify
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, labels=[0, 1],
                                       target_names=["No", "Yes"], zero_division=0)

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
        fig, ax = plt.subplots(figsize=(18, 9))
        plot_tree(pipeline.named_steps["tree"], feature_names=feature_names,
                  class_names=["No", "Yes"], filled=True, rounded=True,
                  max_depth=4, fontsize=8, ax=ax)
        ax.set_title("Decision Tree (display limited to depth 4 for readability)")
        fig.tight_layout()

        MODEL.update({"pipeline": pipeline, "features": X.columns.tolist(), "target": target_col,
                      "mapping": {str(negative_label): "No", str(positive_label): "Yes"},
                      "tree_feature_names": feature_names, "classes": classes})
        metrics = (f"Training complete\nAccuracy: {acc:.2%}\nRows used: {len(df)}\n"
                   f"Target column: {target_col}\nOriginal class mapping: {negative_label} → No | {positive_label} → Yes\n"
                   f"Features: {len(X.columns)}")
        return metrics, report, fig, f"Model ready. Enter feature values as JSON using these keys: {X.columns.tolist()}"
    except Exception as e:
        return f"Could not train model: {e}", "", None, "Fix the issue above and train again."


def predict_one(input_json):
    try:
        if MODEL["pipeline"] is None:
            return "Please upload a dataset and train the model first."
        data = json.loads(input_json)
        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object/dictionary.")
        missing = [c for c in MODEL["features"] if c not in data]
        if missing:
            raise ValueError(f"Missing feature(s): {missing}")
        row = pd.DataFrame([{c: data[c] for c in MODEL["features"]}])
        pred = int(MODEL["pipeline"].predict(row)[0])
        probability = MODEL["pipeline"].predict_proba(row)[0]
        confidence = float(probability[pred])
        answer = "YES" if pred == 1 else "NO"
        return f"Prediction: {answer}\nModel confidence: {confidence:.1%}"
    except Exception as e:
        return f"Prediction error: {e}"


def load_example():
    example = {"Age": 30, "Monthly_Income": 45000, "Credit_Score": 720,
               "Employment_Status": "Employed", "Loan_Amount": 200000, "Existing_Loan": "No"}
    return json.dumps(example, indent=2)

with gr.Blocks(title="Generic Decision Tree Yes/No Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Decision Tree ML — Yes/No Predictor\nUpload your own CSV, select a binary target column, train a Decision Tree, view its graph, then enter new feature values to get a **Yes/No** decision.")
    with gr.Tab("1. Upload & Train"):
        csv_file = gr.File(label="Upload your dataset (.csv)", file_types=[".csv"], type="filepath")
        inspect_btn = gr.Button("Preview Dataset")
        preview = gr.Dataframe(label="First 10 rows", interactive=False)
        dataset_info = gr.Textbox(label="Dataset information", lines=5, interactive=False)
        target = gr.Dropdown(label="Target column (what you want to predict)", choices=[], interactive=True)
        with gr.Row():
            test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.05, label="Test data fraction")
            max_depth = gr.Slider(1, 20, value=5, step=1, label="Decision Tree max depth")
            min_leaf = gr.Slider(1, 10, value=1, step=1, label="Minimum samples per leaf")
        train_btn = gr.Button("Train Decision Tree", variant="primary")
        metrics = gr.Textbox(label="Model results", lines=6, interactive=False)
        report = gr.Textbox(label="Classification report", lines=7, interactive=False)
        tree_plot = gr.Plot(label="Decision Tree graph")
        predict_hint = gr.Textbox(label="Prediction instructions", interactive=False)
        inspect_btn.click(inspect_dataset, inputs=csv_file, outputs=[preview, target, dataset_info])
        train_btn.click(train_model, inputs=[csv_file, target, test_size, max_depth, min_leaf],
                        outputs=[metrics, report, tree_plot, predict_hint])
    with gr.Tab("2. Predict Yes / No"):
        gr.Markdown("After training, enter **every feature column** as a JSON object. Use the exact column names from your CSV. Numeric values should be numbers; category values should be text.")
        sample_btn = gr.Button("Load example input (for the included sample dataset)")
        input_box = gr.Code(label="Feature values in JSON format", language="json", value="{}")
        predict_btn = gr.Button("Predict Yes / No", variant="primary")
        prediction = gr.Textbox(label="Decision", lines=3, interactive=False)
        sample_btn.click(load_example, outputs=input_box)
        predict_btn.click(predict_one, inputs=input_box, outputs=prediction)
    gr.Markdown("**Important:** For target columns other than explicit Yes/No, the app maps the two classes alphabetically: first class → No, second class → Yes. Check the mapping shown after training. This is a learning demo, not a real-world approval system.")

demo.launch()
