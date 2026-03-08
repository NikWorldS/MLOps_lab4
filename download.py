import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/NikWorldS/MLOps_lab4/refs/heads/main/car_price_prediction.csv', delimiter = ',')
    df.to_csv("cars.csv", index = False)
    return df

def preprocessing_data_frame(path_to_csv):
    df = pd.read_csv(path_to_csv)
    cat_columns = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Color']
    num_columns = ['ID', 'Price', 'Levy', "Prod. year", 'Leather interior', 'Engine volume', 'Turbo engine', "Mileage", "Cylinders", 'Wheel', "Airbags"]

    # перевод пробега в float
    df['Mileage'] = (
        df['Mileage']
        .str.replace(" km", "")
        .str.replace(",", ".")
        .astype(float)
    )

    # Перевод флага "Leather interior" в int флаг
    df["Leather interior"] = (
        df["Leather interior"] == "Yes"

    ).astype(int)

    # Перевод "Engine volume" в два разных столбца
    df[["Engine volume", "Turbo_feature"]] = df["Engine volume"].str.split(" ", n=1, expand=True)
    df["Engine volume"] = df["Engine volume"].astype(float)
    df["Turbo_feature"] = (df["Turbo_feature"] == "Turbo").astype(int)

    # Перевод "Whell" в int флаг
    df["Wheel"] = (df["Wheel"] == "Left wheel").astype(int)

    # Перевод "Levy" в int
    df["Levy"] = df["Levy"].str.replace("-", "0").astype(float)

    # Отсеивание строк по году
    suspicious_years = df[df["Prod. year"] < 1971]
    df = df.drop(suspicious_years.index)

    # Отсеивание по пробегу (слишком малый)
    suspicious_mileage = df[(df["Prod. year"] < 2020) & (df["Mileage"] < 1100)]
    df = df.drop(suspicious_mileage.index)
    # Отсеивание по пробегу (слишком большой)
    suspicious_mileage = df[(df["Mileage"] > 1e6)]
    df = df.drop(suspicious_mileage.index)

    # Отсеивание по объёму двигателя (слишком малый)
    suspicious_engine = df[df["Engine volume"] < 0.4]
    df = df.drop(suspicious_engine.index)
    # Отсеивание по объёму двигателя (слишком большой)
    suspicious_engine = df[df["Engine volume"] > 6]
    df = df.drop(suspicious_engine.index)

    # Отсеивание по цене (слишком малой)
    suspicious_price = df[df["Price"] < 5000]
    df = df.drop(suspicious_price.index)
    # Отсеивание по цене (слишком большой)
    suspicious_price = df[df["Price"] > 10000]
    df = df.drop(suspicious_price.index)

    df = df.reset_index(drop=True)
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv("processed_cars.csv", index = False)
    return df

download_data()
preprocessing_data_frame("cars.csv")