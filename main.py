from huggingface_hub import notebook_login
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
    
)


import io
from fastapi import UploadFile, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


peft_model_id = "Elkhayyat17/qlora-code-llama2"
config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    use_auth_token=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, peft_model_id)


def agent(inputs):
  
  format_template = "If you are a doctor, please answer the medical questions based on the patient's description.\n{questation}\n\n ans"

  # First, format the prompt
#questation = "A 45-year-old woman comes to the clinic with a persistent cough, fever, and difficulty breathing. On examination, you notice decreased breath sounds on one side of the chest. What are the possible differential diagnoses, and what initial steps should be taken in the evaluation and management of this patient?."
#prompt = format_template.format(questation=questation)
  ask_samantha = '''
  Symptoms:
  '''

  messages = [{"role": "system", "content": '''You are Doctor Sakenah, a virtual AI doctor known for your friendly and approachable demeanor,
  combined with a deep expertise in the medical field. You're here to provide professional, empathetic, and knowledgeable advice on health-related inquiries.
  You'll also provide differential diagnosis. If you're unsure about any information, Don't share false information.'''},
  {"role": "user", "content": f" Symptoms:{inputs}"}]
  prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Inference can be done using model.generate

  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
  with torch.autocast("cuda", dtype=torch.float16):
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.2,
    )


  out = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
  start_index = out.find("[/INST]")

# Extract the portion starting from [/INST]
  desired_output = out[start_index+7:]


  return desired_output







class ChatBotRequest(BaseModel):
    message: str

@app.post("/chatBot")
async def detect(request: ChatBotRequest):
    return { "response":  agent(request.message) }


