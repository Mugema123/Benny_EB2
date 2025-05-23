from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import split_features_labels, train_test_split_data
from src.models.train_model import train_xgboost_model, evaluate_model
from src.visualization.visualize import plot_feature_importance

df = load_data("data/raw/school_data.csv")
df = preprocess_data(df)
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

model = train_xgboost_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

plot_feature_importance(model)
