from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import gradio as gr
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Initialize the Hugging Face model pipeline
pipe = pipeline(task="text-classification", model="Rahul13/my_awesome_model")

# Request body structure for FastAPI
class TextInput(BaseModel):
    text: str

# FastAPI route for prediction
@app.post("/predict")
async def predict_category(input_data: TextInput):
    try:
        # Get the prediction from the model
        result = pipe(input_data.text)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function for Gradio to classify text using the same pipeline
def classify_text(input_text):
    result = pipe(input_text)
    return result[0]["label"], result[0]["score"]

# Gradio Interface creation
interface = gr.Interface(
    fn=classify_text, 
    inputs="text", 
    outputs=["label", "number"],
    title="Text Classification",
    description="Classify text into predefined categories"
)

# Route for accessing the Gradio interface
@app.get("/")
async def gradio_interface():
    return {"message": "Go to /gradio for the Gradio interface"}

# Launch Gradio on a separate thread, or launch within FastAPI
@app.on_event("startup")
async def launch_gradio():
    interface.launch(inline=False, server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

# Start the app via uvicorn if this script is run directly
