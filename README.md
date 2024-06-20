#  Resume Parsing and Analysis:
This application is built for employers looking for candidates against a particular job description. The model extracts the  key information (name, education, work experience, skills) from a sample resumes. The algorithm maps the  candidate resumes with job descriptions based on the extracted information. A similarity score given the resume of the candidate and a job description.



## Features:
1. **Resume Analysis:** Users can upload resumes into the system. The system then processes this textual data, extracting relevant information and generating embeddings to represent the semantic meaning of the text.
2. **Semantic Matching:** The system computes the similarity between the embeddings of resumes and job descriptions using techniques like cosine similarity. This allows the system to quantitatively measure the degree of alignment between a candidate's qualifications and the requirements outlined in the job description.
3. **Integration with Generative AI Model:** The system is integrated with a Generative AI model (Gemini Model) to provide more comprehensive feedback and recommendations. This model can generate responses based on predefined prompts, offering insights into missing keywords, profile summaries, and recommended courses/resources.

## Modules:
1. **Text Embedding Module:** This module is responsible for transforming textual data into numerical representations, facilitating analysis and comparison. It leverages state-of-the-art NLP models such as [BERT](https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b) and [Doc2Vec] (https://cs.stanford.edu/~quocle/paragraph_vector.pdf)to convert raw text into high-dimensional embeddings. BERT, precisely [transformer-based model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), captures contextualized embeddings that encode the semantic meaning of words and sentences. On the other hand, Doc2Vec generates fixed-size vectors representing entire documents, enabling efficient comparison of resumes and job descriptions. By utilizing these techniques, the Text Embedding Module ensures that textual information is translated into a format suitable for further analysis and evaluation within the application.

2. **Cosine Similarity Module:** The [Cosine Similarity](https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity) Module computes the similarity between pairs of embeddings using the cosine similarity metric. This metric measures the cosine of the angle between two vectors, providing a measure of their alignment in a high-dimensional space. By comparing the embeddings of resumes and job descriptions, this module determines the degree of match between a candidate's qualifications and the requirements of a job. 

3. **Generative AI (Gemini)**: The application integrates with [Generative AI models](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens), specifically the Gemini model, to provide additional insights and recommendations for resume improvement. Through the Gemini model, the application can generate content and recommendations based on user inputs, enhancing the overall utility of the system.



## System Design:
<img src = "images\system_design.jpg">

## Data Flow:
<img src = "images\data_flow.jpg">


## Interface:
<img src = "images\interface_1.jpg">
<img src = "images\interface_2.jpg">
<img src = "images\interface_3.jpg">
<img src = "images\interface_4.jpg">

## Usage:
1. Clone the repository .
```bash
    git clone https://github.com/farvath/Resume-Parser-and-Analysis.git
```
2. Create a python virtual environment, activate it and install the packages from `requirements.txt` file :
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file and place the GOOGLE_API_KEY [Link](https://aistudio.google.com/app/apikey). 

4. Run the `app.py`
``` streamlit run app.py```






