"""Package a trained model as a Nano model folder (the model contract,
docs/model-contract.md):

    export/<model-id>/
        model.onnx     the exported network
        model.json     decoder family, input size, labels, provenance
        labels.txt     class names, one per line, in class-id order

host/deploy_model.sh copies the folder to the Nano and builds its TensorRT
engine there (an engine can only be built on the device that runs it).
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from cvtrain.export import decoder_from_shapes, onnx_signature
from cvtrain.hparams import DECODER
from cvtrain.paths import EXPORT_DIR


def package_model(*, onnx_path, arch, img_size, class_names, model_id,
                  metrics=None, source_note="", out_root=EXPORT_DIR) -> Path:
    in_shape, outputs = onnx_signature(onnx_path)
    decoder, n_classes = decoder_from_shapes(outputs)

    # The checks that would otherwise only fail on the Nano, much later:
    if decoder != DECODER[arch]:
        raise RuntimeError(f"{arch} exported an ONNX that looks like a {decoder} "
                           f"head (outputs {outputs}); expected {DECODER[arch]}")
    if in_shape[-1] != img_size:
        raise RuntimeError(f"ONNX input is {in_shape}, expected {img_size}x{img_size}")
    if n_classes != len(class_names):
        raise RuntimeError(f"ONNX predicts {n_classes} classes but the dataset "
                           f"has {len(class_names)} names")

    out_dir = Path(out_root) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out_dir / "model.onnx")
    (out_dir / "labels.txt").write_text("\n".join(class_names) + "\n")

    source = (f"{arch} fine-tuned on a PC with training/ ({date.today().isoformat()}), "
              f"exported to ONNX opset 12 at {img_size}x{img_size}.")
    if metrics:
        source += " Validation (best epoch): mAP50-95 {:.3f}, mAP50 {:.3f}.".format(
            metrics["map50_95"], metrics["map50"])
    if source_note:
        source += " " + source_note
    meta = {
        "label": f"{model_id} ({arch} {img_size})",
        "task": "detect",
        "decoder": decoder,
        "input_size": img_size,
        "labels": "labels.txt",
        "onnx": "model.onnx",
        "engine": "model.fp16.engine",
        "source": source,
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir
