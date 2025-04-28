# HCMUS_Capydata: Sport Action Image Captioning

## Project Overview
This project focuses on developing an image captioning model for sports-related images. The goal is to generate accurate and descriptive captions for various sports events, athletes, and key moments using deep learning techniques. All captions are written in Vietnamese.

## Objectives
1. Collect and curate a dataset of sports images.
2. Preprocess and annotate the dataset with descriptive captions in Vietnamese.
3. Develop an image captioning model using deep learning.
4. Evaluate model performance and improve caption quality.
5. Deploy the model for real-world applications in sports analytics and media.

## Data Collection and Annotation
- Source: Images were automatically crawled from Pinterest using the keyword format: "sport + 'action in real life'."
- Annotation Process: Captions were generated using OpenAI API and later reviewed and refined by native Vietnamese speakers with sports expertise.

## Project Organization
```
📂 CapyData_ImageCaptioning
```

## Getting Started
To set up and run this project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/mtue04/Capydata-SportIC.git
cd Capydata-SportIC
```

2. Set up the environment:
```bash
conda env create --file capydata-env.yml
conda activate capydata-env
```

3. Run the notebooks:
- Open the Jupyter notebooks in the `notebooks/` folder to view and run model training and evaluation steps.
- Run file `app/app.py` to initialize our website to generate Vietnamese captions.

## Evaluation Metrics
- Data Quality Metrics:
    - Caption length distribution
    - Vocabulary diversity
    - Word frequency analysis
    - Image quality checks
- Model Performance Metrics:
    - BLEU, ROUGE (n-gram-based)
    - BERTScore (semantic similarity)

## Modeling Approaches

Four different approaches were considered for training the model:

1. Traditional Pipeline:
- CNN + LSTM (Show and Tell model)

2. Attention-Based Pipeline:
- CNN + LSTM + Attention (Show, Attend and Tell)

3. Transformer-Based Pipeline:
- CNN/ViT + Transformer (e.g., BART, T5)

4. Modern Pretrained Models:
- Encoder: CLIP, Oscar, ViT
- Decoder: GPT/Vision-Language Transformers
- Fine-tuning models like BLIP-2 or multimodal architectures

## How to Contribute

If you want to contribute to this project, follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Create a pull request with a detailed explanation of your changes.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it with proper attribution.

## Group's Resources (Timeline, Meeting notes, Docs)
- [Project Drive](https://drive.google.com/drive/u/0/folders/1bOzHSAkUOj7Zp-7Flf7NnK-F441BHmwe).

## Contributors
This project is done by the Capydata group, University of Science, VNU-HCM.

| Student ID | Name                   |
|------------|------------------------|
| 22127234   | Cao Hoàng Lộc          |
| 22127360   | Võ Nguyễn Phương Quỳnh |
| 22127440   | Phan Võ Minh Tuệ       |
| 22127450   | Phạm Anh Văn           |