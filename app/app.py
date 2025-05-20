import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import json
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load Vietnamese Embedding model
embed_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

# Load vocabulary early to avoid NameError
vocab_path = "data/capydata_ic/vocab/vocab.json"
idx_to_word_path = "data/capydata_ic/vocab/idx_to_word.json"
if not os.path.exists(vocab_path) or not os.path.exists(idx_to_word_path):
    logger.error(f"Vocabulary file missing: {vocab_path} or {idx_to_word_path}")
    raise FileNotFoundError("Vocabulary files not found")

logger.info("Loading vocabulary...")
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = {k: int(v) for k, v in json.load(f).items()}
with open(idx_to_word_path, "r", encoding="utf-8") as f:
    idx_to_word = {int(k): v for k, v in json.load(f).items()}
vocab_size = len(vocab)
logger.info(f"Loaded vocab, size: {vocab_size}")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, encoded_size=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())
        for layer in modules[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.resnet = nn.Sequential(*modules[:-2])
        self.fc = nn.Linear(2048, encoded_size)
        self.bn = nn.BatchNorm1d(encoded_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha = torch.softmax(att, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


# Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, encoder_dim=256, attention_dim=256
    ):
        super().__init__()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropoutOthers = nn.Dropout(0.6)
        self.dropoutLSTM = nn.Dropout(0.6)
        self.hidden_size = hidden_size

    def forward(self, encoder_out, captions, teacher_forcing_ratio=0.5):
        batch_size = encoder_out.size(0)
        vocab_size = self.fc.out_features
        embeddings = self.embedding(captions)
        embeddings = self.dropoutLSTM(embeddings)
        h, c = torch.zeros(batch_size, self.hidden_size).to(
            encoder_out.device
        ), torch.zeros(batch_size, self.hidden_size).to(encoder_out.device)
        outputs = torch.zeros(batch_size, captions.size(1), vocab_size).to(
            encoder_out.device
        )

        for t in range(captions.size(1)):
            context, _ = self.attention(encoder_out.unsqueeze(1), h)
            input_lstm = torch.cat([embeddings[:, t], context], dim=1)
            input_lstm = self.dropoutLSTM(input_lstm)
            h, c = self.lstm(input_lstm, (h, c))
            h = self.dropoutOthers(h)
            outputs[:, t] = self.fc(h)
        return outputs


# Combined Model
class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        encoded_size=256,
        embed_size=256,
        hidden_size=512,
        vocab_size=vocab_size,
        attention_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.encoder = EncoderCNN(encoded_size=encoded_size)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_dim=encoded_size,
            attention_dim=attention_dim,
        )
        self.device = device
        self.to(device)

    def forward(self, images, captions, teacher_forcing_ratio=0.5):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, teacher_forcing_ratio)
        return outputs

    @torch.no_grad()
    def generate_caption(self, image, vocab, idx_to_word, max_length=20, beam_width=5):
        self.eval()
        image = image.to(self.device)
        features = self.encoder(image).unsqueeze(1)
        sequences = [[[], 0.0, [features, None]]]
        for _ in range(max_length):
            all_candidates = []
            for seq, score, state in sequences:
                if seq and seq[-1] == vocab["</s>"]:
                    all_candidates.append([seq, score, state])
                    continue
                token_id = vocab["<s>"] if not seq else seq[-1]
                token = torch.tensor([[token_id]], device=self.device)
                embed = self.decoder.embedding(token)
                decoder_hidden = (
                    state[1][0]
                    if state[1]
                    else torch.zeros(1, self.decoder.hidden_size, device=self.device)
                )
                context, _ = self.decoder.attention(state[0], decoder_hidden)
                input_lstm = torch.cat([embed.squeeze(1), context], dim=1)
                if state[1] is None:
                    h, c = self.decoder.lstm(input_lstm)
                else:
                    h, c = self.decoder.lstm(input_lstm, state[1])
                output = self.decoder.fc(h)
                output = torch.softmax(output, dim=-1)
                top_probs, top_indices = output.topk(beam_width)
                for i in range(beam_width):
                    next_token = top_indices[0, i].item()
                    penalty = 0.1 * sum(1 for t in seq if t == next_token)
                    next_seq = seq + [next_token]
                    next_score = score + torch.log(top_probs[0, i]).item() - penalty
                    all_candidates.append([next_seq, next_score, [features, (h, c)]])
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
                :beam_width
            ]
        captions = []
        for seq, score, _ in sequences:
            caption = [
                idx_to_word.get(idx, "<unk>")
                for idx in seq
                if idx not in [vocab["<s>"], vocab["</s>"]]
            ]
            captions.append((" ".join(caption), score))
        return captions


# Vocabulary class
class ImageCaptionVocab:
    def __init__(self, vocab_path, idx_to_word_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        self.vocab = {k: int(v) for k, v in vocab_json.items()}
        with open(idx_to_word_path, "r", encoding="utf-8") as f:
            idx_to_word_json = json.load(f)
        self.idx_to_word = {int(k): v for k, v in idx_to_word_json.items()}
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    def get_vocab(self):
        return self.vocab


# Image transformation
def process_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


# Caption post-processing
def process_caption(caption):
    caption = caption.replace("<unk>", "").replace("_", " ")
    caption = " ".join(caption.split())
    if caption:
        caption = caption[0].upper() + caption[1:]
    if caption and caption[-1] not in [".", "!", "?"]:
        caption += "."
    return caption


# Load models
def load_models(
    vocab_path="data/capydata_ic/vocab/vocab.json",
    idx_to_word_path="data/capydata_ic/vocab/idx_to_word.json",
    model_path="models/2_cnn_lstm_attention.pth",
    embed_file_path="models/pretrained_embeds.npy",
):
    logger.info("Starting to load models...")

    # Check file existence
    for file_path in [vocab_path, idx_to_word_path, model_path]:
        if not os.path.exists(file_path):
            logger.error(f"File doesn't exist: {file_path}")
            raise FileNotFoundError(f"Couldn't find file: {file_path}")

    # Load vocabulary
    logger.info("Loading vocabulary...")
    try:
        vocab_obj = ImageCaptionVocab(
            vocab_path=vocab_path, idx_to_word_path=idx_to_word_path
        )
        vocab = vocab_obj.get_vocab()
        idx_to_word = vocab_obj.idx_to_word
        vocab_size = len(vocab)
        logger.info(f"Loaded vocab, size: {vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        raise

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    logger.info("Initializing model...")
    model = ImageCaptionModel(
        encoded_size=256,
        embed_size=256,
        hidden_size=512,
        vocab_size=vocab_size,
        attention_dim=256,
        device=device,
    ).to(device)

    # Load or generate Vietnamese embeddings
    logger.info(f"Checking for Vietnamese embeddings at {embed_file_path}...")
    if os.path.exists(embed_file_path):
        logger.info("Loading Vietnamese embeddings from file...")
        try:
            pretrained_embeds = np.load(embed_file_path)
            if pretrained_embeds.shape != (vocab_size, 256):
                logger.error(
                    f"Embedding file shape {pretrained_embeds.shape} does not match expected ({vocab_size}, 256)"
                )
                raise ValueError("Embedding file shape mismatch")
            model.decoder.embedding.weight.data = torch.tensor(
                pretrained_embeds, dtype=torch.float32
            ).to(device)
            logger.info("Embeddings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embed_file_path}: {e}")
            raise
    else:
        logger.info("Embedding file not found. Generating Vietnamese embeddings...")
        pretrained_embeds = np.zeros((vocab_size, 256))
        words = list(vocab.keys())
        batch_size = 128
        for i in range(0, len(words), batch_size):
            batch_words = words[i : i + batch_size]
            try:
                embeds = embed_model.encode(batch_words)
                for word, embed in zip(batch_words, embeds):
                    idx = vocab[word]
                    pretrained_embeds[idx] = embed[:256]
            except Exception as e:
                logger.warning(f"Failed to encode batch starting at index {i}: {e}")
                for word in batch_words:
                    idx = vocab[word]
                    pretrained_embeds[idx] = np.zeros(256)
        os.makedirs(os.path.dirname(embed_file_path), exist_ok=True)
        np.save(embed_file_path, pretrained_embeds)
        logger.info(f"Embeddings generated and saved to {embed_file_path}")
        model.decoder.embedding.weight.data = torch.tensor(
            pretrained_embeds, dtype=torch.float32
        ).to(device)

    model.decoder.embedding.weight.requires_grad = True

    # Load model weights
    logger.info(f"Loading model weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model weights from {model_path}: {e}")
        raise

    model.eval()
    logger.info("Model loaded successfully")
    return model, vocab, idx_to_word, device


# Load models when starting the server
model, vocab, idx_to_word, device = load_models()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received POST request to /predict")
    if "image" not in request.files:
        logger.error("No image found in request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["image"]
    if file.filename == "":
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            logger.info("Starting image processing")
            image_tensor = process_image(file.stream).to(device)
            logger.info(f"Image tensor shape: {image_tensor.shape}")

            logger.info("Starting caption generation")
            with torch.no_grad():
                captions = model.generate_caption(
                    image_tensor, vocab, idx_to_word, max_length=20, beam_width=5
                )
                logger.info(f"Generated {len(captions)} captions")
                processed_captions = [process_caption(cap[0]) for cap in captions]
                for i, cap in enumerate(processed_captions):
                    logger.info(f"Generated caption {i+1}: {cap}")

            return jsonify(
                {
                    "filename": filename,
                    "captions": processed_captions,
                }
            )

        except Exception as e:
            logger.exception(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500

    logger.error(f"Invalid file format: {file.filename}")
    return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
