AI Service Tools:

1. ICD-10-CM code mapping based on patient diagnosis input

2. CPT code mapping based on patient surgery/procedure input

3. AI model that correlates CPT code and ICD-10-CM code, if they are medically relevant to one another.

Requirements:

Please refer to requirements.txt file

To install via requirements.txt run the following:

 pip install -r requirements.txt
 
If CUDA is used for torch run this afterwards:

 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


 cpt_code.py:
   Handles the CPT code mapping, taking the input as patient information(JSON input) and mapping it with the CPT code list csv file. 
 icd_code.py:
   Handles the ICD-10-CM code mapping, taking the input as patient information(JSON input) and mapping it with the ICD-10-CM code list csv file.

 app.py:
  Utilizes a finetuned Llama3 model called m42-health/Llama3-Med42-8B, it is a big model around 20gbs. 
  
  This file has 3 main functions, 
  
   extract_json function: it handles the output, where it keeps only the useful information from the prompt.
   
   normalize_icd function: this function deals with the input icd code, as sometimes it comes formatted differently that our data, so we normalized it for our data.
   
   check_alignment function: The bulk of this file, validates the ICD & CPT codes, then calls the prompt and returns the output.

  All 3 files contain a FastAPI server built in for testing via Postman.
