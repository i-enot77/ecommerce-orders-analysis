"""
Czyszczenie i analiza zamówień e-commerce

Skrypt generuje brudny zbiór danych, wykonuje eksplorację, czyści dane,
dodaje kolumny analityczne, przygotowuje podstawowe agregacje oraz zapisuje
oczyszczony plik CSV.
"""

import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_messy_data(output_path: str = "zamowienia_messy.csv") -> None:
    """Generuje przykładowy plik z brudnymi danymi zamówień."""
    np.random.seed(42)

    n = 500
    klienci = [
        "Anna Kowalska",
        "  Jan Nowak",
        "Anna Kowalska",
        "PIOTR WIŚNIEWSKI",
        "katarzyna lewandowska",
        "Tomasz Zieliński ",
        "Marta Wójcik",
        "anna kowalska ",
        "Krzysztof Kamiński",
        " Magdalena Dąbrowska",
    ]
    produkty = [
        "Laptop",
        "Mysz",
        "Klawiatura",
        "Monitor",
        "laptop",
        "MYSZ",
        "Słuchawki",
        "Pendrive",
        "monitor",
        "Webcam",
    ]
    kategorie = [
        "Elektronika",
        "elektronika",
        "ELEKTRONIKA",
        "Akcesoria",
        "akcesoria",
        "Akcesoria ",
    ]
    miasta = [
        "Warszawa",
        "Kraków",
        "warszawa",
        "Gdańsk",
        "WROCŁAW",
        "Poznań",
        "Łódź ",
        " Warszawa",
        "kraków",
    ]

    start_date = datetime(2025, 1, 1)
    daty_iso = [
        (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in np.random.randint(0, 300, n // 2)
    ]
    daty_pl = [
        (start_date + timedelta(days=int(d))).strftime("%d.%m.%Y")
        for d in np.random.randint(0, 300, n // 2)
    ]
    daty = daty_iso + daty_pl
    np.random.shuffle(daty)

    df = pd.DataFrame(
        {
            "order_id": range(1001, 1001 + n),
            "klient": np.random.choice(klienci, n),
            "produkt": np.random.choice(produkty, n),
            "kategoria": np.random.choice(kategorie, n),
            "miasto": np.random.choice(miasta, n),
            "ilosc": np.random.choice(
                [1, 2, 3, 5, -1, 0],
                n,
                p=[0.5, 0.2, 0.15, 0.1, 0.025, 0.025],
            ),
            "cena_jednostkowa": np.random.choice(
                [
                    "199.99",
                    "299,99",
                    "1 499.00",
                    "89.50",
                    "2999",
                    "399.00 zł",
                    None,
                    "abc",
                ],
                n,
            ),
            "data_zamowienia": daty,
            "email": np.random.choice(
                [
                    "anna@gmail.com",
                    "JAN@WP.PL",
                    "piotr.w@onet",
                    "marta@gmail.com",
                    "tomasz@interia.pl",
                    None,
                    "krzysztof.k@gmail.com",
                    "brak",
                ],
                n,
            ),
        }
    )

    for col in ["miasto", "kategoria", "data_zamowienia"]:
        df.loc[df.sample(frac=0.05, random_state=1).index, col] = np.nan

    df = pd.concat([df, df.sample(20, random_state=2)], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Wygenerowano plik '{output_path}' — {len(df)} wierszy")


def explore_data(df: pd.DataFrame) -> None:
    """Wyświetla podstawowe informacje diagnostyczne o danych."""
    print("\n=== Rozmiar danych ===")
    print(df.shape)

    print("\n=== Informacje o kolumnach ===")
    print(df.info())

    print("\n=== Statystyki opisowe ===")
    print(df.describe(include="all"))

    print("\n=== Liczba braków danych ===")
    print(df.isnull().sum())

    print("\n=== Rozkłady wartości w kolumnach tekstowych ===")
    for column in ["klient", "produkt", "kategoria", "miasto", "email"]:
        print(f"\n--- {column} ---")
        print(df[column].value_counts(dropna=False))


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści dane zamówień i zwraca DataFrame gotowy do analizy."""
    df_clean = df.copy()

    # 1. Usunięcie pełnych duplikatów wierszy.
    df_clean = df_clean.drop_duplicates()

    # 2. Ujednolicenie zapisu tekstu.
    df_clean["klient"] = df_clean["klient"].astype("string").str.strip().str.lower().str.title()
    df_clean["produkt"] = df_clean["produkt"].astype("string").str.strip().str.lower().str.title()
    df_clean["miasto"] = df_clean["miasto"].astype("string").str.strip().str.lower().str.title()
    df_clean["kategoria"] = df_clean["kategoria"].astype("string").str.strip().str.lower()

    # 3. Konwersja dat zapisanych w dwóch różnych formatach.
    df_clean["data_zamowienia"] = pd.to_datetime(
        df_clean["data_zamowienia"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    # 4. Czyszczenie ceny i zmiana typu na liczbowy.
    df_clean["cena_jednostkowa"] = (
        df_clean["cena_jednostkowa"]
        .astype("string")
        .str.strip()
        .str.replace("zł", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df_clean["cena_jednostkowa"] = pd.to_numeric(
        df_clean["cena_jednostkowa"],
        errors="coerce",
    )

    # 5. Usunięcie rekordów bez kluczowych danych.
    df_clean = df_clean.dropna(subset=["cena_jednostkowa", "data_zamowienia"])

    # 6. Uzupełnienie mniej krytycznych braków wartościami technicznymi.
    df_clean["miasto"] = df_clean["miasto"].fillna("Unknown")
    df_clean["kategoria"] = df_clean["kategoria"].fillna("unknown")
    df_clean["email"] = df_clean["email"].fillna("brak_emaila")

    # 7. Usunięcie zamówień z błędną liczbą sztuk.
    df_clean = df_clean[df_clean["ilosc"] > 0].copy()

    return df_clean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumny potrzebne do dalszej analizy."""
    df = df.copy()

    df["wartosc_zamowienia"] = df["ilosc"] * df["cena_jednostkowa"]
    df["rok"] = df["data_zamowienia"].dt.year
    df["miesiac"] = df["data_zamowienia"].dt.month
    df["nazwa_dnia"] = df["data_zamowienia"].dt.day_name()

    email_pattern = r"^[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}$"
    df["email_poprawny"] = df["email"].astype(str).str.match(email_pattern)

    return df


def run_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tworzy podstawowe zestawienia biznesowe."""
    sprzedaz_miesieczna = (
        df.groupby("miesiac", as_index=False)["wartosc_zamowienia"]
        .sum()
        .sort_values("miesiac")
    )

    top_klienci = (
        df.groupby("klient", as_index=False)["wartosc_zamowienia"]
        .sum()
        .sort_values("wartosc_zamowienia", ascending=False)
        .head(5)
    )

    srednia_kategoria = (
        df.groupby("kategoria", as_index=False)["wartosc_zamowienia"]
        .mean()
        .sort_values("wartosc_zamowienia", ascending=False)
    )

    return sprzedaz_miesieczna, top_klienci, srednia_kategoria


def plot_monthly_sales(sprzedaz_miesieczna: pd.DataFrame) -> None:
    """Tworzy i zapisuje wykres miesięcznej wartości zamówień."""
    plt.figure(figsize=(10, 5))
    plt.bar(
        sprzedaz_miesieczna["miesiac"],
        sprzedaz_miesieczna["wartosc_zamowienia"],
    )
    plt.title("Łączna wartość zamówień w poszczególnych miesiącach")
    plt.xlabel("Miesiąc")
    plt.ylabel("Wartość zamówień")
    plt.xticks(sprzedaz_miesieczna["miesiac"])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("wartosc_zamowien_miesiecznie.png", dpi=150)
    plt.show()


def main() -> None:
    generate_messy_data()

    df = pd.read_csv("zamowienia_messy.csv")
    explore_data(df)

    print("\n=== Zidentyfikowane problemy jakościowe ===")
    problems = [
        "zduplikowane wiersze",
        "niespójny zapis tekstu: spacje, wielkość liter i różne warianty tych samych wartości",
        "ceny zapisane jako tekst, z przecinkami, spacjami, walutą lub błędnymi wartościami",
        "daty zapisane w dwóch różnych formatach",
        "braki danych w kilku kolumnach",
        "niepoprawne ilości: wartości zerowe i ujemne",
        "niektóre adresy e-mail nie spełniają podstawowego wzorca poprawności",
    ]
    for number, problem in enumerate(problems, start=1):
        print(f"{number}. {problem}")

    df_clean = clean_orders(df)
    df_clean = add_features(df_clean)

    sprzedaz_miesieczna, top_klienci, srednia_kategoria = run_analysis(df_clean)

    print("\n=== Łączna wartość zamówień w każdym miesiącu ===")
    print(sprzedaz_miesieczna)

    print("\n=== Top 5 klientów według wartości zamówień ===")
    print(top_klienci)

    print("\n=== Średnia wartość zamówienia według kategorii ===")
    print(srednia_kategoria)

    plot_monthly_sales(sprzedaz_miesieczna)

    df_clean.to_csv("zamowienia_clean.csv", index=False)
    print("\nZapisano oczyszczony plik: zamowienia_clean.csv")


if __name__ == "__main__":
    main()
