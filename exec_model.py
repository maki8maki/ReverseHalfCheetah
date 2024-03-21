import hydra
from omegaconf import OmegaConf

from config import Config

@hydra.main(config_path='config/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    data_path = 'logs/Reverse/20240320-2254/episode_1000.pth'
    cfg.executer.dreamer.load(data_path)
    cfg.executer.view()
    cfg.executer.close()

if __name__ == '__main__':
    main()
