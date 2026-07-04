from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_logistic_regression(**kwargs):
    return LogisticRegression(max_iter=1000, random_state=42, **kwargs)

def get_random_forest(**kwargs):
    return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)

def get_decision_tree(**kwargs):
    return DecisionTreeClassifier(random_state=42, **kwargs)

MODELS = {
    "Logistic Regression": get_logistic_regression,
    "Random Forest": get_random_forest,
    "Decision Tree": get_decision_tree,
}
