import os
import gc
import torch
import wandb
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

class Trainer(metaclass=ABCMeta):
    def __init__(self, save_root, accel):
        self.save_root = save_root
        self.accel = accel

    @abstractmethod
    def compute_loss_and_update(self):
        pass
    
    def memory_optimization(self):

        # wait
        self.accel.wait_for_everyone()

        # memory deallocation
        gc.collect()

        # removing cache
        torch.cuda.empty_cache()

        # wait
        self.accel.wait_for_everyone()

    def wait_and_save_ckpt(self, **kwargs):

        model = kwargs['model']
        batch_ind = kwargs['batch_ind']
        length_dataloader = kwargs['epochs'] * len(kwargs['train_dataloader'])
        save_number = kwargs['save_number']
        processor = kwargs['processor']

        # wait for everyone
        self.accel.wait_for_everyone()
        if batch_ind+1 in [int(i/save_number*length_dataloader) for i in range(1, save_number+1)]:
            
            # Student
            unwrapped_model = self.accel.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(self.save_root, f'{batch_ind+1}'),
                is_main_process=self.accel.is_main_process and self.accel.local_process_index==0,
                save_function=self.accel.save,
                state_dict=self.accel.get_state_dict(model),
                max_shard_size='3GB'
                )

            # processor
            processor.save_pretrained(os.path.join(self.save_root, f'{batch_ind+1}'))
            
            # print
            self.accel.print(f"----{batch_ind+1}: Save Comleted!!----")
        
        # wait for everyone
        self.accel.wait_for_everyone()

    def train(self, **kwargs):
        """
        necessary kwargs

        - model
        - vllm_model
        - epochs
        - train_dataloader
        - optimizer
        - scheduler
        - processor
        - max_new_tokens
        - wandb
        - save_number
        """
        for epoch in range(kwargs['epochs']):

            # progress bar
            prog_bar = tqdm(enumerate(kwargs['train_dataloader']),
                            disable=not (self.accel.is_main_process and self.accel.local_process_index==0),
                            total=len(kwargs['train_dataloader']))

            # training start
            for batch_ind, inputs in prog_bar:

                # memory opt
                self.memory_optimization()

                # forward & backward
                with self.accel.accumulate(kwargs['model']):
                    # backwarding loss with gradient accumulation
                    loss_dict = self.compute_loss_and_update(inputs, **kwargs)
                
                # wandb logging
                if kwargs['wandb'] and self.accel.is_main_process and self.accel.local_process_index==0:
                    update_wandb_dict = {'lr': kwargs['scheduler'].get_last_lr()[0]}
                    for k, v in loss_dict.items(): update_wandb_dict.update({k: v})
                    wandb.log(update_wandb_dict)

                # displaying progress bar
                GPU0_usage = torch.cuda.memory_reserved(device=0) / 1024**3
                prog_bar.set_description(f"[GPU0:{GPU0_usage:.0f}][LR:{kwargs['scheduler'].get_last_lr()[0]:.6f}] " +\
                                            " | ".join([f"{k}: {v:.3f}" for k, v in loss_dict.items()]), refresh=True)

                # saving the model
                kwargs['batch_ind'] = epoch * len(kwargs['train_dataloader']) + batch_ind
                self.wait_and_save_ckpt(**kwargs)