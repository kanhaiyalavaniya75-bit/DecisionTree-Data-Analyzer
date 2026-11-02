# Generic Decision Tree Yes/No Predictor

A Gradio web interface for training a Decision Tree classifier on a user-uploaded CSV dataset, viewing the decision-tree graph and getting a Yes/No prediction for new input.

## Features
- Upload your own `.csv` dataset
- Preview dataset and select the target column
- Automatic median imputation for numeric columns and most-frequent imputation + one-hot encoding for categorical columns
- Train/test split and accuracy/classification report
- Decision Tree visualization
- Enter new feature values as JSON and receive **YES** or **NO** plus model confidence

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the local URL printed in the terminal.

## Use in Google Colab
1. Upload `app.py` and `requirements.txt` to Colab, or clone/download this repository.
2. Run `!pip install -r requirements.txt`.
3. Run `!python app.py` (or put the app code in a cell and change the final line to `demo.launch(share=True)`).
4. Upload your CSV through the interface.

## Dataset requirements
- CSV must have at least one feature column and one target column.
- The selected target column must contain exactly **two classes**. For example `Yes/No`, `Approved/Rejected`, or `Pass/Fail`.
- If the target labels are already Yes and No, they are preserved. Otherwise the app maps the two labels alphabetically: first label → No, second label → Yes. Review the mapping displayed after training.
- The app expects a JSON object containing all feature columns for a prediction. Example: `{"Age": 30, "Income": 40000, "Student": "No"}`. Use the exact feature names and suitable value types from your own dataset.

**Educational project only:** predictions are only as useful as the uploaded data and should not be used for real lending, hiring, medical or other high-impact decisions.
