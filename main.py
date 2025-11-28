from fastapi import FastAPI
import gradio as gr
import pickle
import pandas as pd

# 1. Load the trained model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Initialize FastAPI
app = FastAPI()

# 3. Define the Prediction Function
def predict_price(median_income, total_rooms, housing_median_age):
    # Prepare input data matching the training columns
    input_data = pd.DataFrame([[median_income, total_rooms, housing_median_age]], 
                              columns=['median_income', 'total_rooms', 'housing_median_age'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return f"${prediction:,.2f}"

# 4. Create the Gradio Interface
# (Removed the 'theme' argument to fix the error)
io = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(0, 15, label="Median Income (Tens of Thousands)"),
        gr.Number(label="Total Rooms in Block", value=1000),
        gr.Slider(1, 100, label="House Age (Years)", value=30)
    ],
    outputs=gr.Textbox(label="Estimated House Value"),
    title="🏡 California House Price Predictor",
    description="Enter the neighborhood details to estimate median house value."
)

# 5. Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, io, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)