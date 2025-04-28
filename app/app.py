import os
import torch
import json
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import logging

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Feature map 7x7
        self.fc = nn.Linear(2048, encoded_size)
        self.bn = nn.BatchNorm2d(encoded_size)

    def forward(self, images):
        features = self.resnet(images)  # [batch_size, 2048, 7, 7]
        features = features.permute(0, 2, 3, 1)  # [batch_size, 7, 7, 2048]
        features = self.fc(features)  # [batch_size, 7, 7, 256]
        features = self.bn(features.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return features


# Attention mechanism
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1) * encoder_out.size(2)
        encoder_out = encoder_out.view(batch_size, num_pixels, -1)
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


# Decoder RNN với beam search trả về 5 captions
class DecoderRNN(nn.Module):
    def __init__(
        self, embed_size, hidden_size, vocab_size, encoder_dim=256, attention_dim=256
    ):
        super().__init__()
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.6)
        self.hidden_size = hidden_size

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)
        vocab_size = self.fc.out_features
        embeddings = self.embedding(captions)
        h, c = torch.zeros(batch_size, self.hidden_size).to(
            encoder_out.device
        ), torch.zeros(batch_size, self.hidden_size).to(encoder_out.device)
        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(encoder_out.device)

        for t in range(seq_len):
            context, _ = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            h_drop = self.dropout(h)
            outputs[:, t] = self.fc(h_drop)
        return outputs

    def generate(self, features, max_len=20, beam_width=5, temperature=1.5):
        """Generate 5 best captions using Beam Search with temperature"""
        self.eval()
        batch_size = features.size(0)
        device = features.device

        start_token = vocab.get_vocab()["<s>"]
        end_token = vocab.get_vocab()["</s>"]
        beam = [
            (
                0.0,
                [start_token],
                torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device),
            )
        ]

        for _ in range(max_len):
            new_beam = []
            for score, tokens, h, c in beam:
                if tokens[-1] == end_token:
                    new_beam.append((score, tokens, h, c))
                    continue
                input_token = torch.tensor([tokens[-1]]).to(device).unsqueeze(0)
                emb = self.embedding(input_token).squeeze(1)
                context, _ = self.attention(features, h)
                lstm_input = torch.cat([emb, context], dim=1)
                h, c = self.lstm(lstm_input, (h, c))
                output = self.fc(h)
                probs = output.softmax(dim=1) / temperature
                probs, top_ids = probs.topk(beam_width, dim=1)
                for prob, token_id in zip(probs[0], top_ids[0]):
                    new_score = score + torch.log(prob).item()
                    new_tokens = tokens + [token_id.item()]
                    new_beam.append((new_score, new_tokens, h.clone(), c.clone()))

            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

        # Trả về 5 caption tốt nhất
        captions = [tokens for _, tokens, _, _ in beam[:5]]
        return [
            torch.tensor(cap) for cap in captions
        ]  # Chuyển từng caption thành tensor


# Vocabulary class
class ImageCaptionVocab:
    def __init__(self, vocab_path=None):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        self.vocab = {k: int(v) for k, v in vocab_json.items()}
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    def get_vocab(self):
        return self.vocab

    def decode(self, token_ids):
        tokens = [
            self.id2word.get(idx, "<unk>")
            for idx in token_ids
            if idx
            not in (
                self.vocab.get("<pad>"),
                self.vocab.get("<s>"),
                self.vocab.get("</s>"),
            )
        ]
        return " ".join(tokens)


# Image transformation với augmentation nhẹ
def process_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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
def load_models():
    vocab_path = os.path.join("models", "vocab.json")
    encoder_path = os.path.join("models", "encoder_best.pth")
    decoder_path = os.path.join("models", "decoder_best.pth")

    for file_path in [vocab_path, encoder_path, decoder_path]:
        if not os.path.exists(file_path):
            logger.error(f"File doesn't exist: {file_path}")
            raise FileNotFoundError(f"Couldn't find file: {file_path}")

    logger.info("All model files found, starting to load models...")
    global vocab
    vocab = ImageCaptionVocab(vocab_path=vocab_path)
    vocab_size = len(vocab.get_vocab())
    logger.info(f"Loaded vocab, size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    encoded_size = 256
    embed_size = 256
    hidden_size = 512
    attention_dim = 256

    logger.info("Initializing encoder...")
    encoder = EncoderCNN(encoded_size=encoded_size).to(device)

    logger.info("Initializing decoder...")
    decoder = DecoderRNN(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        encoder_dim=encoded_size,
        attention_dim=attention_dim,
    ).to(device)

    logger.info("Loading weights for encoder...")
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(
        encoder_checkpoint["state_dict"]
        if "state_dict" in encoder_checkpoint
        else encoder_checkpoint
    )

    logger.info("Loading weights for decoder...")
    decoder_checkpoint = torch.load(decoder_path, map_location=device)
    decoder.load_state_dict(
        decoder_checkpoint["state_dict"]
        if "state_dict" in decoder_checkpoint
        else decoder_checkpoint
    )

    encoder.eval()
    decoder.eval()
    logger.info("All models loaded successfully")

    return encoder, decoder, vocab, device


# Load models when starting the server
encoder, decoder, vocab, device = load_models()


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
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        logger.info(f"Saving file at: {filepath}")
        file.save(filepath)

        try:
            logger.info("Starting image processing")
            image_tensor = process_image(filepath).to(device)
            logger.info(f"Image tensor shape: {image_tensor.shape}")

            logger.info("Starting caption generation")
            with torch.no_grad():
                logger.info("Extracting features from encoder")
                features = encoder(image_tensor)
                logger.info(f"Features shape: {features.shape}")

                logger.info("Generating captions from decoder")
                generated_ids_list = decoder.generate(
                    features, beam_width=5, temperature=1.5
                )
                logger.info(f"Generated IDs list length: {len(generated_ids_list)}")

                # Decode và xử lý 5 caption
                captions = []
                for i, generated_ids in enumerate(generated_ids_list):
                    caption = vocab.decode(generated_ids.cpu().numpy())
                    caption = process_caption(caption)
                    captions.append(caption)
                    logger.info(f"Generated caption {i+1}: {caption}")

            # Trả về danh sách 5 caption
            return jsonify(
                {"filename": filename, "filepath": filepath, "captions": captions}
            )

        except Exception as e:
            logger.exception(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500

    logger.error(f"Invalid file format: {file.filename}")
    return jsonify({"error": "Invalid file format"}), 400


if __name__ == "__main__":
    app.run(debug=True)
