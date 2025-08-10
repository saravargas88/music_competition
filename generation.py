from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel , Field
import uvicorn
import os
import scipy.io.wavfile
import random 
import torch 
from transformers import pipeline
import traceback 
import pathlib 
import numpy as np


#What does this do? 
'''
Sets up a FastAPI application for generating music using a pretrained model.
It defines an endpoint to generate music based on a text prompt and duration.
FastAPI is a web framework for bulding PAIs with Python based on standarrd Python type hints. 
you use the app object to define the routes and operations of my API
'''
os.environ["TOKENIZERS_PARALLELISM"]= "false" #Disable tokenizers parallelism warning

#---- Global variables ---- 

# Create a FastAPI app instance
app= FastAPI(title= "Music Generation API")

# Global synthesizer variable
synthesizer = None

#define the device to use for model inference
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

#------------ PYDANTIC MODELS ----------
# Field() function is a basemodel function allows to define fields of the class. f
class MusicRequest(BaseModel): #BaseModel class from pydantic 
    prompt: str= Field(..., description= "Text prompt for music generation"),
    duration: int = Field(10, description="Duration should be less than 30 seconds"),
    n_agents: int = Field(4, description= "Number of music generating agents", ge=1, le=10)

    
class TrackResponse( BaseModel) : 
    agent_id:int
    path: str #path to the generated audio file
   
class GenResponses(BaseModel):  #Trying to represent the whole API 
    results: list[TrackResponse]
    
 # --------------- post function -----------   
    
@app.on_event("startup")
async def load_model():
    global synthesizer, device

    #Limit GPU memory usage if using CUDA
    if device.type == "cuda":
        try: 
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
            print("Limited GPU memory usage to 80%")
        except Exception as e:
            print("Could not limit GPU memory usage:", e)
    
    print(f"Loading MusicGen on {device.type}...")
    
    # Load the MusicGen model using the transformers pipeline
    synthesizer = pipeline(
        "text-to-audio",
        model ="facebook/musicgen-small", 
        device =0 if device.type=="cuda" else -1)

    print("Model loaded successfully")
    
@app.get("/")
def health_check():
    
    return {"status": "ok", "device": device.type}
    
#from fastapi.middleware.cors import CORSMiddleware
#app.add_middleware(
 #   CORSMiddleware, 
    


@app.post("/generate", response_model= GenResponses)
#response_model specifies return type
def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    #call synthesize function for each agent
    #save audio script 
    global synthesizer
    
    if synthesizer is None: 
        raise HTTPException(status_code=500, detail="Model not loaded yet")
    
   
    max_new_tokens= request.duration * 50 
    
    results: list[TrackResponse] = []
    
    try: 
        
        for agent_id in range (request.n_agents):
            #I COULD ADD A RANDOM SEED FOR EACH AGENT TO MAKE MORE DIVERSE
            #generate music for each agent 
            try:
                out = synthesizer(
                    request.prompt,
                    generate_kwargs=dict(do_sample=True, max_new_tokens=max_new_tokens),
                )
            except TypeError:
                out = synthesizer(
                    request.prompt,
                    forward_params=dict(do_sample=True, max_new_tokens=max_new_tokens),
                )
            
            # Expect: {"audio": [np.ndarray], "sampling_rate": int}
            try:
                sr = int(out["sampling_rate"])
                audio = out["audio"][0]
                
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                    
                # create file path
                filename= f"agent_{agent_id}".wav
                filepath=pathlib.Path("generated_audios") / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                #save audio to file 
                scipy.io.wavfile.write(filepath, sr, audio)
                
                #save the audio file to this 
                results.append(TrackResponse(agent_id=agent_id, path = str(filepath)))
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Bad pipeline output: {e}")


            

            return GenResponses(results=results)
            #save audio to file 
            

            
    except  Exception as e: 
        raise HTTPException(status_code=500, detail =f"Error generating music : {str(e)}")
    
    
    
     
    