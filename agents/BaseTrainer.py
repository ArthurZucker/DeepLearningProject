import pytorch_lightning as pl
from utils.agent_utils import get_datamodule, get_net
from utils.use_artifact import get_artifact


class BaseTrainer:
    def __init__(self, config, run) -> None:
        self.config = config.hparams
        self.wb_run = run
        self.network_param = config.network_param # used to log artifact's metadata? 
        self.encoder = config.network_param.backbone
        
        if  hasattr(config.network_param,"artifact"):
            config.network_param.weight_checkpoint = get_artifact( config.network_param.artifact )
        
            
        self.model = get_net(
            config.hparams.arch, config.network_param, config.optim_param
        )
        
        self.wb_run.watch(self.model)
        self.datamodule = get_datamodule(
            config.hparams.datamodule, config.data_param,config.hparams.dataset
        )
        
        
        
    
    

    def run(self):

        if self.config.tune_lr:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
                default_root_dir = self.wb_run.save_dir
                
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)

        if self.config.tune_batch_size:
            trainer = pl.Trainer(
                logger=self.wb_run,
                gpus=self.config.gpu,
                auto_scale_batch_size="power",
                accelerator="auto",
                default_root_dir = self.wb_run.save_dir
            )
            trainer.logger = self.wb_run
            trainer.tune(self.model, datamodule=self.datamodule)
    
    
