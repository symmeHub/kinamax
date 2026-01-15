import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl
    import altair as alt
    return alt, mo, pl


@app.cell
def _(pl):
    orbits = pl.read_parquet("outputs/orbits.parquet")
    orbits
    return (orbits,)


@app.cell
def _(alt, mo, orbits):
    chart = mo.ui.altair_chart(alt.Chart(orbits).mark_point().encode(
        x='fd',
        y=alt.Y('Eh').scale(type='log'),
        color='detected_subharmonic'
    ))

    chart
    return


@app.cell
def _(alt, mo, orbits):
    chart4 = mo.ui.altair_chart(alt.Chart(orbits).mark_point().encode(
        x='fd',
        y=alt.Y('Ev').scale(type='log'),
        color='detected_subharmonic'
    ))

    chart4
    return


@app.cell
def _(orbits):
    orbits.group_by("orbit_label").sum()
    return


@app.cell
def _(orbits):
    orbits
    return


@app.cell
def _(orbits, pl):
    unique_orbits = orbits.unique(subset=["orbit_label"], keep="first")["orbit_label", "fd", "detected_subharmonic"]
    orbit_energy = orbits["orbit_label", "Eh"].group_by("orbit_label").sum()
    orbit_energy
    orbit_data = unique_orbits.join(orbit_energy, on="orbit_label", how="inner")
    #orbit_data = orbit_data.with_columns((pl.col("Eh") * pl.col("fd") / pl.col("detected_subharmonic")).alias("Ph"))
    orbit_data = orbit_data.with_columns((pl.col("Eh") * pl.col("fd")).alias("Ph"))
    orbit_data
    return (orbit_data,)


@app.cell
def _(alt, mo, orbit_data):
    chart2 = mo.ui.altair_chart(alt.Chart(orbit_data).mark_point().encode(
        x='fd',
        y=alt.Y('Ph').scale(type='log'),
        color='detected_subharmonic'
    ))

    chart2
    return


@app.cell
def _(alt, orbit_data, pl):
    custom_colors = {
        0: "#BAC24C",  # yellow
        1: "#5179D6",  # orange
        2: "#2ca02c",  # green
        3: "#d62728",  # red
        # 4: "#9467bd",  # purple
        5: "#9651D6",  # brown
    }

    chart3 = (
        alt.Chart(
            orbit_data.with_columns((pl.col("Ph") * 1000.0).alias("Ph_scaled"))
        )
        .mark_circle(size=100)
        .encode(
            x=alt.X("fd:Q").title("Drive Frequency [Hz]"),
            y=alt.Y("Ph_scaled").scale(type="log").title("Harvested Power [mW]"),
            color=alt.Color(
                "detected_subharmonic:N",
                title="Detected subharmonic",
                scale=alt.Scale(
                    domain=list(custom_colors.keys()),
                    range=list(custom_colors.values()),
                ),
            ),
            shape="detected_subharmonic:N",
            tooltip=["orbit_label", "fd", "Ph", "detected_subharmonic"],
        )
        .properties(
            title="Damping dissipation vs. Drive Frequency", width=800, height=400
        )
    ).interactive()

    chart3.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
