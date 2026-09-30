"""Prepare pinned CLIP ONNX encoders and golden fixtures; not used by C++ at runtime.

pip install transformers==4.57.1 onnxruntime==1.23.2 onnx==1.19.1
Uses existing torch/OpenCV. Model weights are downloaded into the requested folder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def pixels(image):
    h, w = image.shape[:2]
    size = (224, int(224 * h / w)) if w <= h else (int(224 * w / h), 224)
    resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    x, y = (size[0] - 224) // 2, (size[1] - 224) // 2
    rgb = resized[y:y+224, x:x+224, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.array([.48145466, .4578275, .40821073], dtype=np.float32)) / np.array([.26862954, .26130258, .27577711], dtype=np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()[None])


class Vision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.projection = model.vision_model, model.visual_projection

    def forward(self, pixel_values):
        return self.projection(self.encoder(pixel_values=pixel_values)[1])


class Text(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.projection = model.text_model, model.text_projection

    def forward(self, input_ids, attention_mask):
        return self.projection(self.encoder(input_ids=input_ids, attention_mask=attention_mask)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--images", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise SystemExit("Completed model bundle already exists; choose a new folder.")
    torch.set_num_threads(4)
    tokenizer = CLIPTokenizerFast.from_pretrained(MODEL, revision=REVISION, cache_dir=args.output / "cache")
    tokenizer.save_pretrained(args.output)
    model = CLIPModel.from_pretrained(MODEL, revision=REVISION, cache_dir=args.output / "cache", attn_implementation="eager").eval()
    vision, text = Vision(model).eval(), Text(model).eval()
    sample = tokenizer("a photo of a bus", padding="max_length", max_length=77, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(vision, (torch.zeros(1, 3, 224, 224),), args.output / "vision.onnx", input_names=["pixel_values"], output_names=["embedding"], opset_version=17, dynamo=False)
        torch.onnx.export(text, (sample.input_ids, sample.attention_mask), args.output / "text.onnx", input_names=["input_ids", "attention_mask"], output_names=["embedding"], opset_version=17, dynamo=False)
        prompts = ["a photo of a bus", "A PHOTO OF A CAT!", "we're testing 123", "  café\tCAFÉ\n", "İstanbul'da bir otobüs", "cafe\u0301", "hello 👋 world", "a photo of a chessboard", "a photo of a butterfly", "a photo of fruits", "a photo of a basketball", "a photo of a building"]
        fixtures = {"texts": [], "images": []}
        for prompt in prompts:
            encoded = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            embedding = torch.nn.functional.normalize(text(encoded.input_ids, encoded.attention_mask), dim=-1)[0].tolist()
            fixtures["texts"].append({"text": prompt, "ids": encoded.input_ids[0].tolist(), "mask": encoded.attention_mask[0].tolist(), "embedding": embedding})
        for index, path in enumerate(args.images):
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {path}")
            name = f"reference-{index}.png"
            cv2.imencode(".png", image)[1].tofile(args.output / name)
            embedding = torch.nn.functional.normalize(vision(pixels(image)), dim=-1)[0].tolist()
            fixtures["images"].append({"file": name, "source": str(path), "embedding": embedding})
    (args.output / "reference.json").write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")
    files = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ["vision.onnx", "text.onnx", "tokenizer.json"]}
    manifest = {"model": MODEL, "revision": REVISION, "dimension": 512, "context_length": 77, "preprocessing": "opencv-cubic-short224-floor-center-rgb-clip-v1", "files_sha256": files, "license": "MIT", "torch": torch.__version__, "opencv": cv2.__version__}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {args.output}; SHA256 manifest and PyTorch reference vectors written.")


if __name__ == "__main__":
    main()
