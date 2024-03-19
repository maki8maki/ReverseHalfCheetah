import hydra
from omegaconf import OmegaConf

from config import Config

@hydra.main(config_path='config/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    cfg.executer.learn(**cfg.learn_config)
    cfg.executer.close()

if __name__ == '__main__':
    main()
