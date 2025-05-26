import streamlit as st
from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import split_features_labels, train_test_split_data
from src.models.train_model import train_xgboost_model, evaluate_model
from src.visualization.visualize import plot_feature_importance
from src.llm.llm_consultant import consult_llm_with_metrics
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🎓 School Dropout Prediction Dashboard")

# Load and preprocess data
df = load_data("data/raw/school_data.csv")
df = preprocess_data(df)

st.subheader("1️⃣ Dataset Preview")
st.dataframe(df.head())

# Feature and label split
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Train model
model = train_xgboost_model(X_train, y_train)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)

# Display metrics
st.subheader("2️⃣ Model Evaluation Metrics")
for key, value in metrics.items():
    st.metric(label=key.capitalize(), value=f"{value:.2f}")

# Feature importance
st.subheader("3️⃣ Feature Importance")
# fig = plot_feature_importance(model, return_fig=True)
# st.pyplot(fig)

# LLM Insights
st.subheader("4️⃣ LLM Recommendations Based on Model Metrics")
if st.button("💡 Get LLM Insights"):
    llm_response = consult_llm_with_metrics(metrics, X.columns.tolist())
    st.text_area("LLM Recommendations", llm_response, height=400)




# def plot_feature_importance(model, return_fig=False):
#     import matplotlib.pyplot as plt
#     importances = model.feature_importances_
#     features = model.get_booster().feature_names
#     sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
#     plt.figure(figsize=(10, 6))
#     plt.barh([features[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
#     plt.xlabel("Feature Importance")
#     plt.title("Top Features")

#     if return_fig:
#         fig = plt.gcf()
#         plt.close(fig)
#         return fig
#     else:
#         plt.show()
