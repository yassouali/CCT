import os, json, math, logging, sys, datetime
import torch
from torch.utils import tensorboard
from utils import helpers
from utils import logger
import utils.lr_scheduler
from utils.htmlwriter import HTML

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

class BaseTrainer:
    def __init__(self, model, resume, config, iters_per_epoch, val_logger=None, train_logger=None):
        self.model = model
        self.config = config

        self.val_logger = val_logger
        self.train_logger = train_logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.do_validation = self.config['trainer']['val']
        self.start_epoch = 1
        self.improved = False

        # SETTING THE DEVICE
        self.device, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        self.model = torch.nn.DataParallel(self.model, device_ids=availble_gpus)
        self.model.to(self.device)

        # CONFIGS
        cfg_trainer = self.config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']

        # OPTIMIZER
        trainable_params = [{'params': filter(lambda p:p.requires_grad, self.model.module.get_other_params())},
                            {'params': filter(lambda p:p.requires_grad, self.model.module.get_backbone_params()), 
                            'lr': config['optimizer']['args']['lr'] / 10}]

        self.optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
        model_params = sum([i.shape.numel() for i in list(model.parameters())])
        opt_params = sum([i.shape.numel() for j in self.optimizer.param_groups for i in j['params']])
        assert opt_params == model_params, 'some params are missing in the opt'

        self.lr_scheduler = getattr(utils.lr_scheduler, config['lr_scheduler'])(optimizer=self.optimizer, num_epochs=self.epochs, 
                                        iters_per_epoch=iters_per_epoch)

        # MONITORING
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
            self.early_stoping = cfg_trainer.get('early_stop', math.inf)

        # CHECKPOINTS & TENSOBOARD
        date_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        run_name = config['experim_name']
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], run_name)
        helpers.dir_exists(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)
         
        writer_dir = os.path.join(cfg_trainer['log_dir'], run_name)
        self.writer = tensorboard.SummaryWriter(writer_dir)
        self.html_results = HTML(web_dir=config['trainer']['save_dir'], exp_name=config['experim_name'],
                            save_name=config['experim_name'], config=config, resume=resume)

        if resume: self._resume_checkpoint(resume)

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            self.logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
            
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus



    def train(self):
        for epoch in range(self.start_epoch, self.epochs+1):
            
            train_results = self._train_epoch(epoch)
            train_results.update((k, str(v)) for k, v in train_results.items())
            
            if self.do_validation and epoch % self.config['trainer']['val_per_epochs'] == 0:
                val_results = self._valid_epoch(epoch)
                self.logger.info('\n\n')
                for k, v in val_results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')
                    
                val_results.update((k, str(v)) for k, v in val_results.items())
                val_log = {'epoch' : epoch, **val_results}
                self.val_logger.add_entry(val_log)
            
            if self.train_logger is not None:
                train_log = {'epoch' : epoch, **train_results}
                self.train_logger.add_entry(train_log)

            # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min': self.improved = (float(val_log[self.mnt_metric]) < self.mnt_best)
                    else: self.improved = (float(val_log[self.mnt_metric]) > self.mnt_best)
                except KeyError:
                    self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break
                    
                if self.improved:
                    self.mnt_best = float(val_log[self.mnt_metric])
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)
                
            #SAVE TRAINING AND VALIDATION RESULTS
            self._save_train_val_logs(self.train_logger, self.val_logger)
        self.html_results.save()
        
    def _save_train_val_logs(self, train_logger, val_logger):
        epoch_results = {"train_results": train_logger.entries,
                         "val_results": val_logger.entries
        }
      
        filename = os.path.join(self.checkpoint_dir, f'train_val_results.json')
        self.logger.info(f'\nSaving epoch results: {filename} ...')
        with open(filename, "w") as f:
            json.dump(epoch_results, f,indent=4)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = os.path.join(self.checkpoint_dir, f'checkpoint.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...') 
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f'Error when loading: {e}')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if "logger" in checkpoint.keys():
            self.train_logger = checkpoint['logger']
        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _valid_epoch(self, epoch):
        raise NotImplementedError

    def _eval_metrics(self, output, target):
        raise NotImplementedError
