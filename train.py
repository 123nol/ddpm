from types import SimpleNamespace
from cond_diffusion import Diffusion
from utils import get_cifar

config = SimpleNamespace(    
    run_name = "cifar10_ddpm_conditional",
    epochs = 25,
    noise_steps=1000,
    seed = 42,
    batch_size = 128,
    img_size = 32,
    num_classes = 10,
    dataset_path = get_cifar(img_size=32),
    train_folder = "train",
    val_folder = "test",
    device = "cuda",
    slice_size = 1,
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    lr = 5e-3)

diff = Diffusion(noise_steps=config.noise_steps , img_size=config.img_size)
diff.prepare(config)
diff.fit(config)