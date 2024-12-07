import wandb

if __name__ == "__main__":
    print(f"Generated WandB run ID: {wandb.util.generate_id()}")
