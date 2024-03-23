import hydra
import torch as th
from omegaconf import OmegaConf

from config import Config

@hydra.main(config_path='config/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    if cfg.w_env_model:
        state_dict = th.load(cfg.model_name, map_location=cfg.executer.dreamer.device)
        cfg.executer.encoder.load_state_dict(state_dict['encoder'])
        cfg.executer.rssm.load_state_dict(state_dict['rssm'])
        cfg.executer.obs_model.load_state_dict(state_dict['obs_model'])
        cfg.executer.action_model.load_state_dict(state_dict['action_model'])
    cfg.executer.learn(**cfg.learn_kwargs)
    cfg.executer.close()

if __name__ == '__main__':
    main()
