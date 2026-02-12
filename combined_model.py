import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Type

from sentence_transformers import SentenceTransformer

from adapters import BaseAdapter, MoEAdapter


class CombinedEmbeddingModel(nn.Module):

    def __init__(
        self,
        base_model: SentenceTransformer,
        adapter: BaseAdapter,
        freeze_base_model: bool = True,
    ):

        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

        if freeze_base_model:
            for p in self.base_model.parameters():
                p.requires_grad = False
            # 防止 BatchNorm 等层在训练时更新统计量
            self.base_model.eval()

    def forward(
        self,
        sentences: Union[str, List[str]],
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False,
        device: Union[str, torch.device, None] = None,
    ) -> torch.Tensor:

        if device is None:
            device = next(self.adapter.parameters()).device
        if isinstance(device, str):
            device = torch.device(device)

        with torch.no_grad():
            base_embs = self.base_model.encode(
                sentences,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
                device=device,
            )

        base_embs = base_embs.detach().clone()

        adapted = self.adapter(base_embs)
        adapted = F.normalize(adapted, p=2, dim=1)

        if not convert_to_tensor:
            return adapted.cpu().numpy()
        return adapted

    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_tensor: bool = True,
        normalize_embeddings: bool = False,
        device: Union[str, torch.device, None] = None,
    ):

        return self.forward(
            sentences,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )

    def save_adapter(self, path: str) -> None:

        torch.save(self.adapter.state_dict(), path)
        print(f"Adapter saved to {path}")

    @classmethod
    def load_from_paths(
        cls,
        base_model_path: str,
        adapter_path: str,
        adapter_type: str | None = None,
        adapter_class: Type[BaseAdapter] | None = None,
        adapter_kwargs: dict | None = None,
        device: str = "cuda",
        freeze_base_model: bool = True,
    ) -> "CombinedEmbeddingModel":

        base_model = SentenceTransformer(base_model_path, device=device)

        state_dict = torch.load(adapter_path, map_location=device)
        input_dim = base_model.get_sentence_embedding_dimension()

        adapter_kwargs = adapter_kwargs or {}

        intermediate_size = adapter_kwargs.get("intermediate_size", 4096)
        num_local_experts = adapter_kwargs.get("num_local_experts", 32)
        num_experts_per_tok = adapter_kwargs.get("num_experts_per_tok", 6)
        hidden_act = adapter_kwargs.get("hidden_act", "silu")
        router_jitter_noise = adapter_kwargs.get("router_jitter_noise", 0.1)
        router_loss_type = adapter_kwargs.get("router_loss_type", "std")

        adapter = MoEAdapter(
            input_dim=input_dim,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            router_jitter_noise=router_jitter_noise,
            router_loss_type=router_loss_type,
        ).to(device)

        adapter.load_state_dict(state_dict)
        return cls(base_model=base_model, adapter=adapter, freeze_base_model=freeze_base_model)