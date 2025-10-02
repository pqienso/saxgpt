from torch.optim.lr_scheduler import (
    StepLR,
    ReduceLROnPlateau,
    LambdaLR
)
import math


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr / optimizer.defaults['lr'],
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """Cosine warmup then cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler from config."""
    if 'scheduler' not in config['training']:
        return None
    
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'cosine':
        warmup_steps = scheduler_config.get('warmup_steps', 0)
        min_lr = scheduler_config.get('min_lr', 0)
        total_steps = scheduler_config.get('total_steps', num_training_steps)
        
        return get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr
        )
    
    elif scheduler_type == 'linear':
        warmup_steps = scheduler_config.get('warmup_steps', 0)
        min_lr = scheduler_config.get('min_lr', 0)
        total_steps = scheduler_config.get('total_steps', num_training_steps)
        
        return get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr
        )
    
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'plateau':
        patience = scheduler_config.get('patience', 5)
        factor = scheduler_config.get('factor', 0.5)
        min_lr = scheduler_config.get('min_lr', 1e-7)
        
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
