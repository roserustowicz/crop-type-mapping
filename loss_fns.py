"""

File to house the loss functions we plan to use.

"""

from keras import losses, optimizers

def get_loss_fn(model_name):
    return losses.categorical_crossentropy

def get_optimizer(optimizer_name, lr, momentum, lrdecay):
    if optimizer_name == "sgd":
        return optimizers.SGD(lr=lr, momentum=momentum, decay=lrdecay)
    elif optimizer_name == "adam":
        return optimizer.Adam(lr=lr, decay=lrdecay)

    raise ValueError(f"Optimizer: {optimizer_name} unsupported")
