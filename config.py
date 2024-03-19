import dacite
import dataclasses
import hydra
from omegaconf import OmegaConf

import dreamer
from utils import set_seed

@dataclasses.dataclass
class Config:
    basename: str
    executer: dreamer.Executer = dataclasses.field(default=None)
    seed: dataclasses.InitVar[int] = None
    learn_config: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self, seed):
        if seed is not None:
            set_seed(seed)
    
    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.executer = hydra.utils.instantiate(_cfg._executer)
        return cfg
