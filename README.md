# tinyml_w4a4
Final project for TinyML (6.5940) in Fall 2024.

Demo of OPT-1.3B quantized with SQ-G32, attn only:

https://github.com/user-attachments/assets/3613a577-4fc3-4da2-b77f-9fac85c1031c

To run the demo script, you can run the following lines in a Google Colab notebook:
```
%cd /content
!rm -r tinyml_w4a4
!git clone https://github.com/lshrack/tinyml_w4a4.git
%cd /content/tinyml_w4a4
!./demo.sh
```

The script will ask for a HuggingFace token. If running the Llama experiments, this must be set to a HuggingFace token with Llama access (to [this model](https://huggingface.co/meta-llama/Llama-3.2-1B)), but for OPT, this does not need to be correctly set to anything. The script will then ask which model, quantization method or methods, and group sizes you would like to try. The results should look like this (this is for OPT with all quantization methods and a group size of 32):

<img width="631" alt="image" src="https://github.com/user-attachments/assets/d3b95059-a0be-483e-84b6-c30c9d34bf48" />



