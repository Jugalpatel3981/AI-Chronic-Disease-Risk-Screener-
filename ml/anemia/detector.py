"""Train and export the automatic nail / skin detector for the anemia check.

Usage (from the repository root; a CUDA GPU makes it much faster):
    python -m ml.anemia.detector               # cross-validation, then final model + export
    python -m ml.anemia.detector --final-only  # final model + export only

Model: torchvision Faster R-CNN with a MobileNetV3-Large FPN backbone
(COCO-pretrained), fine-tuned to find two kinds of box: nail and skin.
Training data: the 750 nail and 750 skin boxes drawn on the 250 photos.

Evaluation (5-fold cross-validation grouped by photo session, so the detector
is always tested on sessions it never saw):
  - box accuracy: share of hand-drawn nails matched by a detection (IoU >= 0.5)
  - end to end: anemia AUROC when the ridge model (trained on hand-drawn boxes
    of the training folds) is fed the *detected* boxes of the test fold,
    compared with feeding it the hand-drawn boxes
The final detector is trained on all photos and exported to ONNX.
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from ml.tabular.config import DOCS_DIR, REPO_ROOT

from .data import load
from .preprocess import batch
from .train import ridge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = REPO_ROOT / "backend" / "models" / "anemia"
LABELS = {1: "nail", 2: "skin"}
EPOCHS, SEED, FOLDS = 30, 0, 5
SCORE_MIN = 0.5


def make_model():
    m = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", min_size=640, max_size=800)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 3)
    return m


def targets(row) -> dict:
    boxes = [b for b in row["nail_xyxy"]] + [b for b in row["skin_xyxy"]]
    labels = [1] * len(row["nail_xyxy"]) + [2] * len(row["skin_xyxy"])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)}


def augment(img: torch.Tensor, tgt: dict, rng) -> tuple[torch.Tensor, dict]:
    """Horizontal flip + mild brightness change. No hue change: colour is the signal."""
    tgt = {k: v.clone() for k, v in tgt.items()}
    if rng.random() < 0.5:
        w = img.shape[2]
        img = img.flip(2)
        tgt["boxes"][:, [0, 2]] = w - tgt["boxes"][:, [2, 0]]
    return TF.adjust_brightness(img, float(rng.uniform(0.85, 1.15))), tgt


def train(rows) -> torch.nn.Module:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = make_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    data = [(TF.to_tensor(r["image"]), targets(r)) for r in rows]
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(len(data))
        for start in range(0, len(order), 2):  # batches of 2 images
            imgs, tgts = zip(*[augment(*data[i], rng) for i in order[start:start + 2]])
            loss = sum(model([i.to(DEVICE) for i in imgs],
                             [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def detect(model, image: np.ndarray) -> tuple[list, list]:
    out = model([TF.to_tensor(image).to(DEVICE)])[0]
    return pick(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy())


def pick(boxes, labels, scores):
    """Top-3 nails and top-3 skin patches above the score threshold."""
    def top(label):
        m = (labels == label) & (scores >= SCORE_MIN)
        order = np.argsort(-scores[m])[:3]
        return [tuple(int(v) for v in b) for b in boxes[m][order]]
    return top(1), top(2)


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def crops(image, boxes):
    return [image[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes if x2 - x1 > 4 and y2 - y1 > 4]


def features(sess, image, nail_boxes, skin_boxes) -> np.ndarray:
    emb = lambda cs: sess.run(None, {"image": batch(cs)})[0].mean(axis=(2, 3)).mean(0)  # noqa: E731
    n, s = emb(crops(image, nail_boxes)), emb(crops(image, skin_boxes))
    return np.concatenate([n, n - s])


def export(model, path, images: list[np.ndarray]) -> None:
    """Trace with a real photo. A random image yields zero detections, and the
    tracer then bakes a wrong shape into the box post-processing."""
    model = model.cpu().eval()
    sample = TF.to_tensor(images[0])
    torch.onnx.export(model, ([sample],), str(path), input_names=["image"],
                      output_names=["boxes", "labels", "scores"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {1: "h", 2: "w"}, "boxes": {0: "n"},
                                    "labels": {0: "n"}, "scores": {0: "n"}})
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for img in images:  # parity on real photos, including ones not used for tracing
        x = TF.to_tensor(img)
        with torch.no_grad():
            ref = model([x])[0]
        boxes, labels, scores = sess.run(None, {"image": x.numpy()})
        keep = ref["scores"].numpy() >= SCORE_MIN
        assert (scores >= SCORE_MIN).sum() == keep.sum(), "ONNX detector output differs from PyTorch"
        ok = np.allclose(boxes[scores >= SCORE_MIN], ref["boxes"].numpy()[keep], atol=1.0)
        assert ok, "ONNX detector boxes differ from PyTorch"


def main(argv: list[str]) -> int:
    final_only = "--final-only" in argv  # skip the cross-validation (results already saved)
    import onnx  # noqa: F401  (needed for the export at the end; fail now, not after training)
    df = load()
    from PIL import Image
    from .data import RAW_DIR
    meta = {int(r.PATIENT_ID): r for r in __import__("pandas").read_csv(RAW_DIR / "metadata.csv").itertuples()}
    to_xyxy = lambda cell: [[x1, y1, x2, y2] for y1, x1, y2, x2 in json.loads(cell)]  # noqa: E731
    rows = []
    for r in df.itertuples():
        m = meta[r.patient_id]
        rows.append({"image": np.asarray(Image.open(RAW_DIR / "photo" / f"{r.patient_id}.jpg").convert("RGB")),
                     "nail_xyxy": to_xyxy(m.NAIL_BOUNDING_BOXES), "skin_xyxy": to_xyxy(m.SKIN_BOUNDING_BOXES),
                     "hb": r.hb, "session": r.session})
    if final_only:
        return train_and_export(rows)
    y = np.array([r["hb"] for r in rows])
    sess = ort.InferenceSession(str(OUT_DIR / "nail_backbone.onnx"), providers=["CPUExecutionProvider"])
    X_gt = np.stack([features(sess, r["image"], r["nail_xyxy"], r["skin_xyxy"]) for r in rows])

    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    pred_gt, pred_det = np.zeros(len(rows)), np.full(len(rows), np.nan)
    matched, total, no_nail = 0, 0, 0
    for fold, (tr, te) in enumerate(cv.split(X_gt, (y < 120).astype(int), groups=[r["session"] for r in rows])):
        print(f"[detector] fold {fold + 1}/{FOLDS}: training on {len(tr)} photos", flush=True)
        det = train([rows[i] for i in tr])
        head = ridge().fit(X_gt[tr], y[tr])
        pred_gt[te] = head.predict(X_gt[te])
        for i in te:
            nails, skins = detect(det, rows[i]["image"])
            for g in rows[i]["nail_xyxy"]:
                total += 1
                matched += any(iou(g, n) >= 0.5 for n in nails)
            if not nails or not skins:
                no_nail += 1
                continue
            pred_det[i] = head.predict(features(sess, rows[i]["image"], nails, skins)[None])[0]
        del det
        torch.cuda.empty_cache()

    ok = ~np.isnan(pred_det)
    results = {
        "nail_recall_iou50": matched / total,
        "photos_without_detection": int(no_nail),
        "auroc_hb_lt_120_handdrawn_boxes": float(roc_auc_score(y < 120, -pred_gt)),
        "auroc_hb_lt_120_detected_boxes": float(roc_auc_score(y[ok] < 120, -pred_det[ok])),
        "auroc_hb_lt_120_handdrawn_same_photos": float(roc_auc_score(y[ok] < 120, -pred_gt[ok])),
        "mae_gl_detected_boxes": float(np.mean(np.abs(y[ok] - pred_det[ok]))),
        "n_photos": len(rows),
    }
    print(json.dumps(results, indent=2))
    (DOCS_DIR / "anemia" / "detector.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return train_and_export(rows)


def train_and_export(rows) -> int:
    print("[detector] training final detector on all photos", flush=True)
    final = train(rows)
    path = OUT_DIR / "nail_detector.onnx"
    export(final, path, [rows[i]["image"] for i in (0, 1, 50, 120, 249)])
    print(f"[detector] exported {path.name} ({path.stat().st_size / 1e6:.1f} MB), ONNX == PyTorch on 5 photos")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
