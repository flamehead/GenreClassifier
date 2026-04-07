import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from db_utils import get_data, get_canadian_data


def eda(canadian: bool = False):
    if canadian:
        df = get_canadian_data()
    else:
        df = get_data()

    genre_counts = df["genre_tzanetakis"].value_counts()
    print(df.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(genre_counts.index, genre_counts.values, color="steelblue")

    # Add numeric labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # x position — center of bar
            height + 50,                          # y position — just above bar
            f"{int(height):,}",                   # label — formatted with comma
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.title("Log Scale Genre Distribution")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Log Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(f"./images/{'canadian_' if canadian else ''}genre_cnt.png", bbox_inches="tight", dpi=150)
    plt.close()



    features = [
        "mood_acoustic", "mood_aggressive", "mood_electronic",
        "mood_happy", "mood_party", "mood_relaxed", "mood_sad",
        "danceability", "gender", "timbre", "tonal", "instrumental",
        "mirex_passionate", "mirex_cheerful", "mirex_melancholy",
        "mirex_aggressive", "mirex_calm"
    ]

    dfMelted = df[features + ['genre_tzanetakis']].melt(
        id_vars = 'genre_tzanetakis',
        var_name = 'feature',
        value_name = 'value'
    )

    g = sns.FacetGrid(data = dfMelted, col="feature",
        col_wrap=4,
        height=4,
        sharey=False
    )

    g.map_dataframe(
        sns.violinplot,
        x="genre_tzanetakis",
        y="value",
        order=sorted(df["genre_tzanetakis"].unique()),
        palette="muted",
        inner="box"
    )

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Genre", "Value")

    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.savefig(f"./images/{'canadian_' if canadian else ''}genre_dist.png", bbox_inches="tight", dpi=150)
    plt.close()


    corr = df[features].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8}
    )

    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(f"./images/{'canadian_' if canadian else ''}feature_corr.png", bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":
    eda(True)