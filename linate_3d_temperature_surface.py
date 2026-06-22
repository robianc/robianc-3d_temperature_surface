from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Coordinate Milano Linate / LIML
LAT = 45.4333
LON = 9.2833

START_YEAR = 1950
END_YEAR = 2026

REQUESTED_END_DATE = date(END_YEAR, 12, 31)

# Open-Meteo/ERA5 può avere qualche giorno di ritardo.
# Così lo script funziona anche se il 2026 non è ancora completo.
AVAILABLE_SAFE_END_DATE = date.today() - timedelta(days=5)
END_DATE = min(REQUESTED_END_DATE, AVAILABLE_SAFE_END_DATE)

OUTDIR = Path(".")
CSV_OUT = OUTDIR / "linate_temperature_daily_1966_2026.csv"
HTML_OUT = OUTDIR / "linate_3d_temperature_surface_1966_2026.html"


def fetch_open_meteo_daily() -> pd.DataFrame:
    """
    Scarica la temperatura media giornaliera da Open-Meteo Historical API,
    usando ERA5.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": f"{START_YEAR}-01-01",
        "end_date": END_DATE.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Rome",
        "models": "era5",
        "temperature_unit": "celsius",
    }

    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)

    print("Scarico dati da:")
    print(url)

    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "daily" not in payload:
        raise RuntimeError(f"Risposta API non valida: {payload}")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(payload["daily"]["time"]),
            "tmean_c": payload["daily"]["temperature_2m_mean"],
        }
    )

    return df


def build_surface_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Costruisce la matrice Z:
    - X = giorno dell'anno, da 1 a 365
    - Y = anno, da 1966 a 2026
    - Z = temperatura media giornaliera

    Il 29 febbraio viene escluso, cosi ogni anno usa lo stesso calendario di 365 giorni.
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month_day"] = df["date"].dt.strftime("%m-%d")

    # Giorni completi di un anno bisestile, così il 29 febbraio ha una colonna propria
    reference_days = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    all_month_days = reference_days.strftime("%m-%d")

    years = list(range(START_YEAR, END_YEAR + 1))

    matrix = df.pivot(index="year", columns="month_day", values="tmean_c")
    matrix = matrix.reindex(index=years, columns=all_month_days)

    z = matrix.to_numpy(dtype=float)

    x = np.arange(1, len(all_month_days) + 1)
    y = np.array(years)

    month_labels = ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
                    "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]

    return x, y, z, month_labels


def month_tick_positions() -> list[int]:
    """
    Posizioni sull'asse X all'inizio di ogni mese,
    usando un calendario non bisestile.
    """
    month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    return [(d - pd.Timestamp("2001-01-01")).days + 1 for d in month_starts]


def plot_3d_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, month_labels: list[str]) -> None:
    """
    Crea superficie 3D interattiva:
    - altezza = temperatura
    - colore = temperatura
    """
    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Turbo",
                colorbar=dict(
                    title="Temperatura<br>[°C]",
                    thickness=18,
                    len=0.75,
                ),
                cmin=np.nanpercentile(z, 1),
                cmax=np.nanpercentile(z, 99),
                hovertemplate=(
                    "Giorno anno: %{x}<br>"
                    "Anno: %{y}<br>"
                    "Temperatura: %{z:.1f} °C"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=(
            f"Milano Linate – temperatura media giornaliera "
            f"({START_YEAR}–{END_YEAR})<br>"
            f"<sup>ERA5/Open-Meteo, coordinate {LAT:.4f}, {LON:.4f}; "
            f"dati fino al {END_DATE.isoformat()}, 2026 parziale</sup>"
        ),
        scene=dict(
            xaxis=dict(
                title="Mese",
                tickmode="array",
                tickvals=month_tick_positions(),
                ticktext=month_labels,
                backgroundcolor="white",
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title="Anno",
                tickmode="linear",
                tick0=1965,
                dtick=5,
                backgroundcolor="white",
                gridcolor="lightgray",
            ),
            zaxis=dict(
                title="Temperatura media giornaliera [°C]",
                backgroundcolor="white",
                gridcolor="lightgray",
            ),
            camera=dict(
                eye=dict(x=1.7, y=-1.8, z=0.9)
            ),
            aspectratio=dict(x=2.2, y=1.4, z=0.55),
        ),
        width=1200,
        height=850,
        margin=dict(l=20, r=20, t=90, b=20),
    )

    fig.write_html(HTML_OUT, include_plotlyjs="cdn")
    print(f"Salvato grafico 3D interattivo: {HTML_OUT}")


def main() -> None:
    df = fetch_open_meteo_daily()

    df = df.loc[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].copy()

    df.to_csv(CSV_OUT, index=False)
    print(f"Salvato CSV: {CSV_OUT} ({len(df)} righe)")

    x, y, z, month_labels = build_surface_matrix(df)

    missing = int(np.isnan(z).sum())
    total = int(z.size)
    print(f"Celle mancanti nella matrice: {missing}/{total}")

    plot_3d_surface(x, y, z, month_labels)


if __name__ == "__main__":
    main()
