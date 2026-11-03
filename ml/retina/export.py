"""Export the 5 fold models as one ONNX ensemble for the API.

Usage (after `python -m ml.retina.train`):
    python -m ml.retina.export

The graph takes a batch of images (N, 3, 384, 384) and returns
  score (N,)       mean predicted DR grade (continuous) across the 5 models
  cam   (N, h, w)  class-activation map: the classifier weights applied to the
                   last feature map. With global-average pooling and a linear
                   head this is exactly how much each location adds to the score.

Two parity checks run before anything is saved:
  1. ONNX output == PyTorch output on real test images (random noise is not
     used: it drives the network to extreme activations where float rounding
     differences between runtimes are amplified)
  2. the API's own code path (raw image -> backend preprocessing -> ONNX)
     reproduces the test-set scores reported by train.py
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import timm
import torch
from scipy.stats import pearsonr

from ml.tabular.config import REPO_ROOT

from .data import GRADE_NAMES, load
from .preprocess import INPUT_SIZE
from PIL import Image

from .train import EVAL_TF, MEAN, STD, WORK_DIR

OUT_DIR = REPO_ROOT / "backend" / "models" / "retina"


class CamModel(torch.nn.Module):
    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        scores, cams = [], []
        for m in self.models:
            f = m.forward_features(x)                       # (N, C, h, w)
            head = m.get_classifier()                       # Linear(C, 1)
            scores.append(head(f.mean(dim=(2, 3))).squeeze(1))
            cams.append(torch.einsum("nchw,c->nhw", f, head.weight[0]))
        return torch.stack(scores).mean(0), torch.stack(cams).mean(0)


def main(backbone: str = "efficientnet_b0.ra_in1k") -> int:
    run_dir = WORK_DIR / backbone.split(".")[0]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    models = []
    for fold in range(5):
        m = timm.create_model(backbone, pretrained=False, num_classes=1)
        m.load_state_dict(torch.load(run_dir / f"fold{fold}.pt", map_location="cpu", weights_only=True))
        models.append(m.eval())
    ensemble = CamModel(models).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = OUT_DIR / "retina.onnx"
    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(ensemble, dummy, str(onnx_path), input_names=["image"], output_names=["score", "cam"],
                      dynamic_axes={"image": {0: "n"}, "score": {0: "n"}, "cam": {0: "n"}}, opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, test = load()
    real = torch.stack([EVAL_TF(Image.open(p).convert("RGB")) for p in test["cached"].iloc[:8]])
    with torch.no_grad():
        ref_s, ref_c = ensemble(real)
    got_s, got_c = sess.run(None, {"image": real.numpy()})
    assert np.allclose(ref_s.numpy(), got_s, atol=1e-3) and np.allclose(ref_c.numpy(), got_c, atol=1e-2), \
        "ONNX export does not match PyTorch"
    print(f"[export] {onnx_path.name}: {onnx_path.stat().st_size / 1e6:.1f} MB, ONNX == PyTorch")

    meta = {
        "onnx": onnx_path.name, "backbone": backbone, "ensemble_size": 5, "input_size": INPUT_SIZE,
        "mean": list(MEAN), "std": list(STD), "cutpoints": results["cutpoints"],
        "calibration": results["calibration"], "referable_threshold": results["referable_threshold"],
        "grade_names": GRADE_NAMES,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Parity check 2: the API's exact code path on raw test images.
    sys.path.insert(0, str(REPO_ROOT))
    import cv2
    from backend.app.models.retina import RetinaService
    svc = RetinaService(OUT_DIR.parent)
    expected = np.load(run_dir / "test_score.npy")
    sample = np.random.default_rng(0).choice(len(test), 40, replace=False)
    got = []
    for i in sample:
        rgb = cv2.cvtColor(cv2.imread(str(test.iloc[i]["path"])), cv2.COLOR_BGR2RGB)
        from ml.retina.preprocess import CACHE_SIZE, prepare
        s, _ = svc.session.run(None, {"image": svc._tensor(prepare(rgb, CACHE_SIZE))})
        got.append(float(np.mean(s)))
    got = np.array(got)
    diff = np.abs(got - expected[sample])
    r = pearsonr(got, expected[sample])[0]
    print(f"[export] API path vs reported test scores: r = {r:.4f}, mean |diff| = {diff.mean():.3f} grade")
    assert r > 0.99, "the API code path does not reproduce the reported predictions"
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
