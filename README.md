AI Service Tools:
1.ICD-10-CM code mapping based on patient diagnosis input
2.CPT code mapping based on patient surgery/procedure input
3.AI model that correlates CPT code and ICD-10-CM code, if they are medically relevant to one another.

Requirements:
Please refer to requirements.txt file

To install via requirements.txt run the following:
 pip install -r requirements.txt
If CUDA is used for torch run this afterwards:
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


 cpt_code.py:
   Handles the CPT code mapping, taking the input as patient information as a JSON input and 
