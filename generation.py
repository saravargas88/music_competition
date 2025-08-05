



from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel 
import uvicorn 
import os
import scipy.io.wavfile
import random 
import torch 
from transformers import pipeline
import traceback 

#What does this do? 
app= FastAPI()

class MusicRequest(BaseModel): #BaseModel class from pydantic 
    prompt: str
    duration: int #duration for each track 
    
os.environ["TOKENIZERS_PARALLELISM"]= "false" #Disable tokenizers parallelism warning

@app.post("/generate-music/")
async def generate_music(request: MusicRequest, background_tasks: BackgroundTasks):
    if request.duration <=0 : 
        raise HTTPException(status_code=400, detail= "Duration must be a positive integer.")
    
    synthesizer= None
    
    try: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {'CUDA' if device.type =='cuda' else 'CPU'}")
        
        '''limiting GPU memory usage: 
            - prevents out of memory errors: dl models use a lot of memory
            - If code tries to use more mem than available it will crash 
            - more stable 
            ''' 
        if device.type =='cuda': 
            try : 
                torch.cuda.set_per_process_memory_fraction(0.8,device=0)
                print("Limited GPU memory usage to 80%")
            except Exception as mem_error: 
                print(f"Failed to limit GPU memory usage: {mem_error}")
                
        #LOAD THE MusicGen MODEL!
        
        synthesizer= pipeline("text-to-audio-generation", model="facebook/musicgen-large", device=0 if device.type=='cuda' else -1 )
        print("Model loaded successfully")
        
        #Generate 4 music generative agents
        results= []
        for agent_id in range(4):
            #set a random seed for each one 
            torch.manual_seed(random.randint(0, 1_000_000))
            print(f"agent {agent_id+1} is generating music")
            
            #synthesizer is a pipeline obj 
            #the first arg is the string prompt 
            #the second arg is the duration in secs
            #max_new_tokens: controls how many audio tokens the model makes (more music time)
            #do_sample=True this enables sampling instead of just greedy next likely token
            #top_k=50: tihs limits sampling to 50  most likely tokens
            #top_p =0.95: limits sampling to the top 95% of  probability mass 
            output = synthesizer(request.prompt, max_new_tokens=request.duration*50, 
                                 do_sample=True, 
                                 top_k=50, 
                                 top_p=0.95, 
                                 temperature=1.0)
            results.append(output)
            
            #return ad process results as needed
            #save the generated audio to a file 
            output_file= f"output_agent_{agent_id}.wav"
            audio_array = output["audio"]
            sample_rate = output["sample_rate"]  
            scipy.io.wavfile.write(output_file, sample_rate, audio_array)
            print(f"Generated music saved to {output_file}")
            
            
            return {"results": results}
        
    except Exception as e: 
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
            
