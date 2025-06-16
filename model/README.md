# Model Folder Contents

This folder contains scripts and notebooks related to the models and experiments used in the paper "Mech Interp - From a Hacker's POV". Below is a brief description of each file:

- **gpt2-custom.py**: Code for loading and running a custom GPT-2 model. The only modification from the original paper is the use of `seed_encodings` to convert the original tokenizer to a seed tokenizer.
- **gpt2-edu_fineweb10B.py**: Script for downloading the EduFineWeb10B dataset, used to train the custom GPT-2 model.
- **gpt2-load-trained.py**: Utility for loading pre-trained GPT-2 models, including custom checkpoints.
- **gpt2-seed_encodings.py**: Script for generating seed encodings for GPT-2 models.
- **gpt2-standard.py**: Code for working with the standard, unmodified GPT-2 model.
- **gpt2.ipynb**: Jupyter notebook for interactive experiments and analyses with GPT-2 models.
- **hellaswag.py**: Script for evaluating models on the HellaSwag benchmark dataset.
- **log/**: Directory containing logs from model training (e.g., `log.txt`).

Each script is documented with comments to explain its purpose and usage. For more details, refer to the code and comments within each file.
