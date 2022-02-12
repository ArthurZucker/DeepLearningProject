import wandb
import os
def get_artifact(name):
    artifact = wandb.run.use_artifact(name, type='model')
    artifact_dir = artifact.download()
    file_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
    return file_path
    