from torch.optim import Optimizer
import pickle
import os
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

from adapters import MoEAdapter
from evaluation import PairwiseEvaluator
from combined_model import CombinedEmbeddingModel
from spcl_loss import SPCLLoss


class TextTripletDataset(Dataset):

    def __init__(
        self,
        samples: list[dict],
    ):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str, str]:
        sample = self.samples[idx]
        return (
            sample["query"],
            sample["pos"],
            sample["neg"],
            sample["neu"],
            sample["label"],
        )


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class AdapterTrainer:
    def __init__(
        self,
        base_model: SentenceTransformer,
        train_path: str,
        eval_path: str,
        device: str = "cuda",
        adapter_kwargs: dict = None,
        eval_label_mode: str = "ternary",
    ):
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        
        input_dim = self.base_model.get_sentence_embedding_dimension()
        
        adapter_kwargs = adapter_kwargs or {}

        intermediate_size = adapter_kwargs.get("intermediate_size", 4096)
        num_local_experts = adapter_kwargs.get("num_local_experts", 32)
        num_experts_per_tok = adapter_kwargs.get("num_experts_per_tok", 6)
        hidden_act = adapter_kwargs.get("hidden_act", "silu")
        router_jitter_noise = adapter_kwargs.get("router_jitter_noise", 0.1)
        router_loss_type = adapter_kwargs.get("router_loss_type", "std")
        self.adapter = MoEAdapter(
            input_dim=input_dim,
            intermediate_size=intermediate_size,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            router_jitter_noise=router_jitter_noise,
            router_loss_type=router_loss_type,
        ).to(self.device)

        self.combined_model = CombinedEmbeddingModel(
            base_model=self.base_model,
            adapter=self.adapter,
            freeze_base_model=True,
        ).to(self.device)

        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        with open(eval_path, "rb") as f:
            eval_data = pickle.load(f)

        self.pairwise_evaluator = PairwiseEvaluator(
            samples=eval_data, device=self.device, label_mode=eval_label_mode
        )

        self.dataset = TextTripletDataset(samples=train_data)

    def train(
        self,
        num_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 3e-3,
        optimizer_type: str = "adamw",
        gate_lr: float = None,
        expert_lr: float = None,
        warmup_steps: int = None,
        use_warmup: bool = True,
        warmup_epochs: float = None,
        max_grad_norm: float = 1.0,
        loss_kwargs: dict | None = None,
        use_router_loss: bool = False,
        router_loss_weight: float = 0.1,
        save_path: str = None,
        eval_epoch: float = 5,
        save_epoch: float = 10
    ) -> None:
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

        optimizer_type = optimizer_type.lower()
        if optimizer_type not in ["adam", "adamw"]:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Supported: 'adam', 'adamw'")

        use_separate_lr = gate_lr is not None and expert_lr is not None
        
        if use_separate_lr:
            gate_lr = gate_lr if gate_lr is not None else learning_rate
            expert_lr = expert_lr if expert_lr is not None else learning_rate
            
            optimizer_params = [
                {'params': self.adapter.moe_layer.gate.parameters(), 'lr': gate_lr},
                {'params': self.adapter.moe_layer.experts.parameters(), 'lr': expert_lr}
            ]
            print(f"Using {optimizer_type.upper()} optimizer with separate learning rates:")
            print(f"  - Gate LR: {gate_lr}")
            print(f"  - Expert LR: {expert_lr}")
        else:
            optimizer_params = self.combined_model.adapter.parameters()
            print(f"Using {optimizer_type.upper()} optimizer with learning rate: {learning_rate}")

        if optimizer_type == "adam":
            if use_separate_lr:
                optimizer = Adam(optimizer_params)
            else:
                optimizer = Adam(optimizer_params, lr=learning_rate)
        else: 
            if use_separate_lr:
                optimizer = AdamW(optimizer_params)
            else:
                optimizer = AdamW(optimizer_params, lr=learning_rate)

        loss_kwargs = loss_kwargs or {}


        spcl_m = loss_kwargs.get("m", 0.25)
        spcl_gamma = loss_kwargs.get("gamma", 256)
        neutral_weight = loss_kwargs.get("neutral_weight", 0.05)
        criterion = SPCLLoss(m=spcl_m, gamma=spcl_gamma, neutral_weight=neutral_weight)

        total_steps = len(dataloader) * num_epochs

        if not use_warmup:
            warmup_steps = 0
            print("Warmup disabled (use_warmup=False).")
        elif warmup_steps is not None:
            print(f"Using specified warmup_steps: {warmup_steps} steps")
        elif warmup_epochs is not None:
            warmup_steps = int(warmup_epochs * len(dataloader))
            print(f"Warmup steps calculated from {warmup_epochs} epochs: {warmup_steps} steps "
                  f"(steps per epoch: {len(dataloader)}, total steps: {total_steps})")
        else:
            warmup_steps = 100
            print(f"Using default warmup_steps: {warmup_steps} steps")
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)

        print("Started Training (online forward).")
        epoch_metrics = [] 
        best_eval_accuracy = None
        best_eval_epoch = None
        best_eval_step = None
        
        eval_by_epoch = eval_epoch >= 1
        save_by_epoch = save_epoch >= 1
        
        steps_per_epoch = len(dataloader)

        if not eval_by_epoch:
            eval_step_interval = max(1, int(eval_epoch * steps_per_epoch))
            print(f"Evaluation mode: step-based (every {eval_step_interval} steps, {eval_epoch*100:.1f}% of epoch)")
        else:
            print(f"Evaluation mode: epoch-based (every {int(eval_epoch)} epochs)")
            
        if not save_by_epoch:
            save_step_interval = max(1, int(save_epoch * steps_per_epoch))
            print(f"Save mode: step-based (every {save_step_interval} steps, {save_epoch*100:.1f}% of epoch)")
        else:
            print(f"Save mode: epoch-based (every {int(save_epoch)} epochs)")
        
        for epoch in range(num_epochs):
            self.combined_model.adapter.train()
            total_loss = 0.0
            
            step_in_epoch = 0

            for batch in dataloader:
                step_in_epoch += 1
                queries, positives, negatives, neutrals, labels = batch

                adapted_q = self.combined_model.encode(
                    list(queries),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
                adapted_p = self.combined_model.encode(
                    list(positives),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
                adapted_n = self.combined_model.encode(
                    list(negatives),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )
                adapted_neu = self.combined_model.encode(
                    list(neutrals),
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    device=self.device,
                )


                sp = (adapted_q * adapted_p).sum(dim=1)
                sn = (adapted_q * adapted_n).sum(dim=1)
                s_neu = (adapted_q * adapted_neu).sum(dim=1)

                with torch.no_grad():
                    base_q = self.base_model.encode(
                        list(queries),
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=self.device,
                    )
                    base_p = self.base_model.encode(
                        list(positives),
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=self.device,
                    )
                    s_orig = (base_q * base_p).sum(dim=1)

                loss = criterion(sp, sn, s_neu, s_orig)

                if use_router_loss and is_moe_adapter:
                    with torch.no_grad():
                        base_q_input = self.base_model.encode(
                            list(queries),
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=self.device,
                        )
                        base_p_input = self.base_model.encode(
                            list(positives),
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=self.device,
                        )
                        base_n_input = self.base_model.encode(
                            list(negatives),
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=self.device,
                        )
                        base_neu_input = self.base_model.encode(
                            list(neutrals),
                            convert_to_tensor=True,
                            normalize_embeddings=True,
                            device=self.device,
                        )
                    moe_inputs = torch.cat(
                        [base_q_input, base_p_input, base_n_input, base_neu_input],
                        dim=0,
                    )
                    _, _, _, router_loss = self.adapter.moe_layer(moe_inputs)
                    loss = loss + router_loss_weight * router_loss

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.adapter.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                
                if not eval_by_epoch:

                    should_eval_step = (step_in_epoch % eval_step_interval == 0) or (step_in_epoch == steps_per_epoch)
                    if should_eval_step:

                        current_avg_loss = total_loss / step_in_epoch

                        eval_results = self.pairwise_evaluator.evaluate(
                            self.combined_model
                        )
                        acc = eval_results["accuracy"]
                        print(
                            f"Evaluation at epoch {epoch+1}, step {step_in_epoch}/{steps_per_epoch}: "
                            f"loss={current_avg_loss:.4f}, accuracy={acc:.4f} (count={eval_results['count']})"
                        )

                        if best_eval_accuracy is None or acc >= best_eval_accuracy:
                            best_eval_accuracy = acc
                            best_eval_epoch = epoch + 1
                            best_eval_step = step_in_epoch
                            if save_path:
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path, exist_ok=True)
                                    print(f"Created save directory: {save_path}")
                                best_model_path = os.path.join(save_path, "best_model_adapter.pth")
                                torch.save(
                                    self.combined_model.adapter.state_dict(),
                                    best_model_path,
                                )

                        epoch_data = {
                            "epoch": epoch + 1,
                            "step": step_in_epoch,
                            "train_loss": current_avg_loss,
                            "eval_accuracy": acc,
                        }
                        epoch_metrics.append(epoch_data)
                
                if save_path and not save_by_epoch:

                    should_save_step = (step_in_epoch % save_step_interval == 0) or (step_in_epoch == steps_per_epoch)
                    if should_save_step:

                        if not os.path.exists(save_path):
                            os.makedirs(save_path, exist_ok=True)
                            print(f"Created save directory: {save_path}")

                        save_filename = f"{save_path}/adapter_at_epoch_{epoch+1}_step_{step_in_epoch}.pth"
                        torch.save(
                            self.combined_model.adapter.state_dict(),
                            save_filename,
                        )
                        print(f"Adapter saved at {save_filename}")

            avg_loss = total_loss / len(dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


            if eval_by_epoch:

                should_eval = (epoch + 1) % int(eval_epoch) == 0
                if should_eval:

                    eval_results = self.pairwise_evaluator.evaluate(
                        self.combined_model
                    )
                    acc = eval_results["accuracy"]
                    print(
                        f"Evaluation after epoch {epoch+1}: "
                        f"accuracy={acc:.4f} (count={eval_results['count']})"
                    )

                    if best_eval_accuracy is None or acc >= best_eval_accuracy:
                        best_eval_accuracy = acc
                        best_eval_epoch = epoch + 1
                        best_eval_step = None  # Epoch-based evaluation (no specific step)
                        if save_path:
                            if not os.path.exists(save_path):
                                os.makedirs(save_path, exist_ok=True)
                                print(f"Created save directory: {save_path}")
                            best_model_path = os.path.join(save_path, "best_model_adapter.pth")
                            torch.save(
                                self.combined_model.adapter.state_dict(),
                                best_model_path,
                            )

                    epoch_data = {
                        "epoch": epoch + 1,
                        "step": None,
                        "train_loss": avg_loss,
                        "eval_accuracy": acc,
                    }
                    epoch_metrics.append(epoch_data)

            if save_path and save_by_epoch:

                should_save = (epoch + 1) % int(save_epoch) == 0
                if should_save:

                    if not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)
                        print(f"Created save directory: {save_path}")

                    save_filename = f"{save_path}/adapter_at_epoch_{epoch+1}.pth"
                    torch.save(
                        self.combined_model.adapter.state_dict(),
                        save_filename,
                    )
                    print(f"Adapter saved at {save_filename}")
        print("Training finished.")

        if best_eval_accuracy is not None:
            if best_eval_step is not None:
                print(
                    f"Best adapter model accuracy={best_eval_accuracy:.4f} "
                    f"at epoch {best_eval_epoch}, step {best_eval_step}. "
                    f"Saved to: best_model_adapter.pth."
                )
            else:
                print(
                    f"Best adapter model accuracy={best_eval_accuracy:.4f} "
                    f"after epoch {best_eval_epoch}. "
                    f"Saved to: best_model_adapter.pth."
                )
        else:
            print("No evaluation was performed during training, so no best adapter model was saved.")
        return epoch_metrics