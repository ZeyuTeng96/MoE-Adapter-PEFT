import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAdapter(nn.Module):


    def __init__(self, input_dim: int):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x


class ExpertMLP(nn.Module):


    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):

        super().__init__()
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = F.silu if hidden_act == "silu" else F.gelu

    def forward(self, hidden_states):

        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MoELayer(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int = 12,
        num_experts_per_tok: int = 3,
        hidden_act: str = "silu",
        router_jitter_noise: float = 0.1,
        router_loss_type: str = "std",
    ):

        super().__init__()
        self.hidden_dim = hidden_size
        self.ffn_dim = intermediate_size
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok
        self.router_loss_type = router_loss_type


        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size, hidden_act) 
            for _ in range(self.num_experts)
        ])
        self.jitter_noise = router_jitter_noise

    def forward(self, hidden_states):
        batch_size, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            hidden_states = hidden_states * noise
        
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if len(top_x) > 0:
                current_state = hidden_states[top_x]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states)


        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights.view(-1)
        importance = torch.zeros(self.num_experts, device=flat_experts.device)
        importance.scatter_add_(0, flat_experts, flat_weights)

        load = torch.zeros(self.num_experts, device=flat_experts.device)
        load.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=routing_weights.dtype))

        if self.router_loss_type == "std":

            router_loss = torch.std(importance)
        elif self.router_loss_type == "switch":

            imp_sum = importance.sum() + 1e-9
            load_sum = load.sum() + 1e-9
            router_loss = (importance * load).sum() * (self.num_experts ** 2) / (imp_sum * load_sum)
        else:
            raise ValueError(f"Unsupported router_loss_type: {self.router_loss_type}. Supported: 'std', 'switch'")

        return final_hidden_states, router_logits, selected_experts, router_loss


class MoEAdapter(BaseAdapter):

    def __init__(
        self,
        input_dim: int,
        intermediate_size: int = 4096,
        num_local_experts: int = 12,
        num_experts_per_tok: int = 3,
        hidden_act: str = "silu",
        router_jitter_noise: float = 0.1,
        router_loss_type: str = "std",
    ):

        super().__init__(input_dim)
        self.moe_layer = MoELayer(
            hidden_size=input_dim,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            router_jitter_noise=router_jitter_noise,
            router_loss_type=router_loss_type,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        updated_embeddings, router_logits, selected_experts, router_loss = self.moe_layer(x)
        updated_embeddings = F.normalize(updated_embeddings, p=2, dim=1)
        return updated_embeddings
