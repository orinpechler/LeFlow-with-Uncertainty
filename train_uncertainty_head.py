"""Train only an uncertainty head on a frozen pretrained LeFlow planner."""

import hydra
from omegaconf import DictConfig

from finetune_for_uncertainty import train


@hydra.main(
    version_base=None,
    config_path="./config/train",
    config_name="uncertainty_head",
)
def run(cfg: DictConfig) -> None:
    train(cfg, head_only=True)


if __name__ == "__main__":
    run()
