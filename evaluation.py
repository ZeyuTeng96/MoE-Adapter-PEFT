import torch
import torch.nn.functional as F
from typing import List

class PairwiseEvaluator:

    def __init__(
        self,
        samples: List[dict],
        device: torch.device,
        label_mode: str = "ternary",
    ):
        self.samples = samples
        self.device = device
        mode = label_mode.lower()
        if mode not in {"binary", "ternary"}:
            raise ValueError(f"label_mode must be 'binary' or 'ternary', got {label_mode}")
        self.label_mode = mode

    def evaluate(
        self,
        model,
        batch_size: int = 1024,
        high_th: float = 0.8,
        low_th: float = 0.2,
        label_mode: str = None,
    ) -> dict:
        model.eval()
        q1_list = [s["q1"] for s in self.samples]
        q2_list = [s["q2"] for s in self.samples]
        labels = [str(s["label"]) for s in self.samples]

        mode = (label_mode or self.label_mode).lower()
        if mode not in {"binary", "ternary"}:
            raise ValueError(f"label_mode must be 'binary' or 'ternary', got {label_mode}")

        preds, labs = [], []
        with torch.no_grad():
            for i in range(0, len(q1_list), batch_size):
                q1_batch = q1_list[i : i + batch_size]
                q2_batch = q2_list[i : i + batch_size]
                lab_batch = labels[i : i + batch_size]

                emb1 = model.encode(
                    q1_batch,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
                emb2 = model.encode(
                    q2_batch,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
                sims = F.cosine_similarity(emb1, emb2)
                for s in sims:
                    score = s.item()
                    if mode == "binary":
                        preds.append("1" if score > high_th else "0")
                    else:
                        if score > high_th:
                            preds.append("1")
                        elif score < low_th:
                            preds.append("0")
                        else:
                            preds.append("2")
                labs.extend(lab_batch)

        correct = sum(p == l for p, l in zip(preds, labs))
        acc = correct / len(labs) if labs else 0.0
        return {"accuracy": acc, "count": len(labs)}