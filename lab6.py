import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def naive_bayes_titanic():
    # Load Titanic dataset and select relevant columns
    df = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare"]].dropna()
    
    # Encode categorical column 'sex'
    df["sex"] = LabelEncoder().fit_transform(df["sex"])
    
    # Define features and target
    X = df[["pclass", "sex", "age", "fare"]]
    y = df["survived"]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))

# Call the function
naive_bayes_titanic()
