import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def get_valid_float(prompt, minimum=0.0, allow_zero=True):
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if value < minimum or (not allow_zero and value == 0):
            print(f"Value must be greater than {minimum}.")
            continue

        return value


def train_and_save_model(csv_path="house_data.csv", model_path="house_price_model.pkl"):
    df = pd.read_csv(csv_path)

    required_columns = ["area", "bedrooms", "bathrooms", "age", "price"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[(df["area"] > 0) & (df["bedrooms"] > 0) & (df["bathrooms"] > 0) & (df["price"] > 0)]

    if df.empty:
        raise ValueError("No valid data rows found. Please check the CSV file.")

    X = df[["area", "bedrooms", "bathrooms", "age"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    joblib.dump(model, model_path)

    print("Model trained successfully!")
    print(f"Model Accuracy: {round(accuracy * 100, 2)}%")
    print(f"Model saved to: {model_path}")
    return model


def predict_house_price(model_path="house_price_model.pkl"):
    model = joblib.load(model_path)

    area = get_valid_float("Enter house area in sq ft: ", minimum=0.0, allow_zero=False)
    bedrooms = get_valid_float("Enter number of bedrooms: ", minimum=0.0, allow_zero=False)
    bathrooms = get_valid_float("Enter number of bathrooms: ", minimum=0.0, allow_zero=False)
    age = get_valid_float("Enter house age in years: ", minimum=0.0, allow_zero=True)

    new_house = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["area", "bedrooms", "bathrooms", "age"],
    )

    predicted_price = model.predict(new_house)[0]
    predicted_price = max(predicted_price, 0)
    print(f"Predicted House Price: ₹{predicted_price:,.2f}")


if __name__ == "__main__":
    train_and_save_model()
    predict_house_price()