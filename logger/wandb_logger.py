import wandb


class WandbLogger:
    def __init__(self, config, model, fold):
        self.cfg_wandb = config["wandb"]

        self.run = wandb.init(
            project=self.cfg_wandb["project_name"],
            entity=self.cfg_wandb["entity"],
            name=self.cfg_wandb["exp_name"] + f"_f{fold}",
            config=config,
        )
        self.run.watch(model)

        self.artifact = wandb.Artifact(
            name=self.cfg_wandb["exp_name"] + f"_f{fold}", type="model"
        )

    def log_info(self, log, epoch):
        self.run.log(log, step=epoch)

    def log_artifact(self, file_path):
        self.artifact.add_file(file_path)
        self.run.log_artifact(self.artifact)

    def finish(self):
        self.run.finish()
