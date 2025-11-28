import pickle

import plotly.express as px
import folium
from fastapi import FastAPI
import gradio as gr
import pandas as pd

OCEAN_PROXIMITY_CHOICES = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
FEATURE_COLUMNS = [
    'median_income',
    'total_rooms',
    'housing_median_age',
    'ocean_proximity',
    'longitude',
    'latitude',
]
TARGET_COLUMN = 'median_house_value'

# Load the trained pipeline once when the service boots
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample data for visualizations so the chart renders instantly during inference
housing_df = pd.read_csv("housing.csv").dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
chart_data = housing_df[[
    'median_income',
    TARGET_COLUMN,
    'ocean_proximity',
]].copy()
if len(chart_data) > 1500:
    chart_data = chart_data.sample(n=1500, random_state=42).reset_index(drop=True)

app = FastAPI()


def build_scatter_plot(median_income: float, predicted_value: float, ocean_proximity: str):
    fig = px.scatter(
        chart_data,
        x='median_income',
        y=TARGET_COLUMN,
        color='ocean_proximity',
        opacity=0.45,
        labels={'median_income': 'Median Income (x $10k)', TARGET_COLUMN: 'Median House Value ($)'},
        title='Median Income vs. House Value',
    )
    fig.add_scatter(
        x=[median_income],
        y=[predicted_value],
        mode='markers',
        marker=dict(size=14, color='red', symbol='x'),
        name='Your Estimate',
    )
    fig.update_layout(legend_title_text='Ocean Proximity', template='plotly_dark')
    return fig


def build_map(latitude: float, longitude: float, price_text: str):
    fmap = folium.Map(location=[latitude, longitude], zoom_start=11)
    folium.Marker([latitude, longitude], tooltip=price_text).add_to(fmap)
    return fmap._repr_html_()


def predict_price(median_income, total_rooms, housing_median_age, ocean_proximity, longitude, latitude):
    input_data = pd.DataFrame([
        [median_income, total_rooms, housing_median_age, ocean_proximity, longitude, latitude]
    ], columns=FEATURE_COLUMNS)

    prediction = model.predict(input_data)[0]
    price_text = f"${prediction:,.2f}"
    scatter_plot = build_scatter_plot(median_income, prediction, ocean_proximity)
    map_html = build_map(latitude, longitude, price_text)
    return price_text, scatter_plot, map_html


examples = [
    [5.1, 2800, 18, "NEAR BAY", -122.23, 37.88],
    [3.4, 4200, 35, "INLAND", -118.45, 34.26],
    [8.0, 1800, 12, "<1H OCEAN", -121.75, 36.77],
]


io = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 15, value=5, step=0.1, label="Median Income (Tens of Thousands)"),
        gr.Number(value=1000, label="Total Rooms in Block"),
        gr.Slider(1, 100, value=30, label="House Age (Years)"),
        gr.Dropdown(choices=OCEAN_PROXIMITY_CHOICES, value="NEAR BAY", label="Ocean Proximity"),
        gr.Slider(-124.5, -113.5, value=-121.5, step=0.01, label="Longitude"),
        gr.Slider(32.0, 42.5, value=37.0, step=0.01, label="Latitude"),
    ],
    outputs=[
        gr.Textbox(label="Estimated House Value"),
        gr.Plot(label="Income vs. Price"),
        gr.HTML(label="Location Map"),
    ],
    title="🏡 California House Price Predictor",
    description=(
        "Enter neighborhood details, including coastal proximity and coordinates, to estimate the median house value."
    ),
    examples=examples,
    cache_examples=False,
)

app = gr.mount_gradio_app(app, io, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)