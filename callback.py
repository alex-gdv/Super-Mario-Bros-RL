from stable_baselines3.common.callbacks import BaseCallback
import os

class SMB_Callback(BaseCallback):
    def __init__(self, model_save_freq, model_name, model_path="./models/", verbose=0):
        super(SMB_Callback, self).__init__(verbose)
        self.model_save_freq = model_save_freq
        self.model_name = model_name
        self.model_path = model_path

    def _init_callback(self):
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.model_save_freq == 0:
            self.model.save(f"{self.model_path}{self.model_name}_{self.num_timesteps}")
        return True
    
    def _on_training_end(self) -> None:
        self.model.save(f"{self.model_path}{self.model_name}_{self.num_timesteps}")
