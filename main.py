from fastapi import FastAPI
import gradio as gr
import pandas as pd
import pickle

OCEAN_PROXIMITY_CHOICES = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

# Load the trained pipeline once when the service boots
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


def predict_price(median_income, total_rooms, housing_median_age, ocean_proximity, longitude, latitude):
    # Construct a single-row frame matching the training column order
    input_data = pd.DataFrame([
        [median_income, total_rooms, housing_median_age, ocean_proximity, longitude, latitude]
    ], columns=[
        'median_income',
        'total_rooms',
        'housing_median_age',
        'ocean_proximity',
        'longitude',
        'latitude',
    ])

    prediction = model.predict(input_data)[0]
    return f"${prediction:,.2f}"


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
    outputs=gr.Textbox(label="Estimated House Value"),
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