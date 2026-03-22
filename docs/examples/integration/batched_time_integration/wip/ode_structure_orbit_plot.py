import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import polars as pl

    return pl, px


@app.cell
def _(pl):
    orbits = pl.read_parquet("data/orbits.parquet")
    orbits
    return (orbits,)


@app.cell
def _(orbits, px):
    fig = px.scatter_3d(
     orbits,
        x="Xa_0",
        z="Xa_1",
        y="fd",
        color=[str(v) for v in orbits["subharmonic"]],
        #size=(orbits["subharmonic"] + 1.0) ** -1 * 0 + 0.05,
        labels={
            "Xa_0": "Position, x",
            "Xa_1": r"Speed, dot x",
            "fd": "Driving frequency, f_d [Hz]",
        },
    )
    fig.update_traces(marker=dict(size=3))

    fig.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
