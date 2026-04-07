import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv("Docker/.env")

CONN_STR = f"host=127.0.0.1 dbname={os.getenv('DB_NAME')} user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')} port={os.getenv('DB_PORT')}"

PULL_COLS = [
        "mood_acoustic", "mood_aggressive", "mood_electronic", "mood_happy", 
        "mood_party", "mood_relaxed", "mood_sad", "danceability", "gender", 
        "timbre", "tonal", "instrumental", "mood_mirex", "genre_tzanetakis"
    ]

def _flatten_mirex(df: pd.DataFrame) -> pd.DataFrame:
    mirex_clusters = [
        "mirex_passionate",   # Cluster 1 — passionate, rousing, confident
        "mirex_cheerful",     # Cluster 2 — rollicking, cheerful, fun
        "mirex_melancholy",   # Cluster 3 — depressed, melancholy, sad
        "mirex_aggressive",   # Cluster 4 — aggressive, tense, angry
        "mirex_calm"          # Cluster 5 — calm, relaxed, serene
    ]

    mirex_df = pd.DataFrame(df["mood_mirex"].tolist())
    mirex_df.columns = mirex_clusters

    return mirex_df.join(df.drop(columns=["mood_mirex"])).dropna()


def get_data() -> pd.DataFrame:
    """Retrieves the music features from Postgres.

    Returns:
        pd.DataFrame: 18 Total Columns
            - 12 Audio features
            - 5 Mirex clusters
            - 1 Genre (target)
    """

    


    query = f"SELECT {', '.join(PULL_COLS)} FROM track_features"

    with psycopg.connect(CONN_STR) as conn:
            df = pd.read_sql_query(query, conn) # type: ignore
    
    return _flatten_mirex(df)


def get_canadian_data() -> pd.DataFrame:
    df_canadian = pd.read_csv('canadian_musicians_and_bands.csv')
    artist_names = [(name) for name in df_canadian["Name"].tolist()]

    with psycopg.connect(CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMPORARY TABLE temp_canadian_artists (name TEXT);")

            with cur.copy("COPY temp_canadian_artists (name) FROM STDIN") as copy:
                for name in artist_names:
                    copy.write_row((name,))

            cur.execute("CREATE INDEX idx_temp_name ON temp_canadian_artists (name);")

        join_query = f"""
            SELECT t.{', t.'.join(PULL_COLS)}
            FROM track_features t
            INNER JOIN temp_canadian_artists c
            ON t.artist = c.name
        """
        
        df = pd.read_sql_query(join_query, conn) # type: ignore

    return _flatten_mirex(df)

