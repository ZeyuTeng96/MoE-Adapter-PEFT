import pickle
from sentence_transformers import SentenceTransformer
from adapter_trainer import AdapterTrainer

TRAIN_PATH = "./data/train.pickle"
EVAL_PATH = "./data/eval.pickle"
EVAL_LABEL_MODE = "ternary" 

#base_model = SentenceTransformer('./e5-mistral-7b-instruct', device="cuda")
base_model = SentenceTransformer('/workspace/embedd_model/multilingual-e5-large', device="cuda")

trainer = AdapterTrainer(
    base_model=base_model,
    train_path=TRAIN_PATH,
    eval_path=EVAL_PATH,
    device="cuda",
    eval_label_mode=EVAL_LABEL_MODE,
    adapter_kwargs={
        "intermediate_size": 4096, 
        "num_local_experts": 12,
        "num_experts_per_tok": 3,
        "hidden_act": "silu", 
        "router_jitter_noise": 0.1, 
        "router_loss_type": "switch", 
    },
)

trainer.train(
    num_epochs=300,
    batch_size=256,
    optimizer_type="adam",
    gate_lr=1e-5,
    expert_lr=1e-4,
    use_warmup=False,
    eval_epoch=1,
    save_epoch=10,
    save_path="./ckpt",
    loss_kwargs={
        "m": 0.25,
        "gamma": 64,
        "neutral_weight": 0.05,
    },
    use_router_loss=False,
)