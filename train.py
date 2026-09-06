from fire import Fire
import yaml
from dm import Unet, GaussianDiffusion, Trainer


def main(
        config_file: str = None,
        # Source dataset at 1024x1024. The training pipeline does a random
        # 512x512 crop with full +/-25% jitter on every step, so ~270 source
        # images produce effectively unlimited unique training patches.
        train_folder: str = './data/diffinfinite-bark_1024',
        # No held-out test set for this run — the eval_loop visualization
        # pulls (deterministically center-cropped) samples from train_folder
        # for snapshot purposes: they are a qualitative progress check,
        # not a generalization measure.
        test_folder: str = None,
        image_size: int = 512,
        dim: int = 256,
        num_classes: int = 4,
        dim_mults: str = '1 2 4',
        channels: int = 4,
        resnet_block_groups: int = 2,
        block_per_layer: int = 2,
        timesteps: int = 1000,
        sampling_timesteps: int = 250,
        batch_size: int = 4,
        # lr: linear scaling rule from upstream (batch 32, lr 1e-4) -> our
        # effective batch 16 -> lr 5e-5. Combined with OneCycleLR's warmup
        # this gives the classic upstream-faithful schedule.
        lr: float = 5e-5,
        train_num_steps: int = 200000,
        save_sample_every: int = 5000,
        gradient_accumulate_every: int = 4,
        save_loss_every: int = 100,
        num_samples: int = 4,
        num_workers: int = 0,
        results_folder: str = './results/bark_1024_aug_200k',
        milestone: int = None,
        # P2 weighting focuses the model's gradient on the perceptually
        # critical content stage (high t / high noise) and away from low-t
        # fine-detail prediction which is the memorization-prone regime.
        # Choi et al. 2022; standard recipe.
        p2_weight_gamma: float = 1.0,
        # Enable/disable image-only color jitter (brightness=0.2, contrast=0.2,
        # saturation=0.1, hue=0.02). Pass `--use_color_jitter False` to disable
        # for fine-tuning from a checkpoint whose colors have drifted.
        use_color_jitter: bool = True,
):

    dim_mults = [int(mult) for mult in dim_mults.split(' ')]

    if config_file:
        with open(config_file, 'r') as config_file:
            config = yaml.safe_load(config_file)
        for key in config.keys():
            locals().update(config[key])

    z_size = image_size // 8

    unet = Unet(
        dim=dim,
        num_classes=num_classes,
        dim_mults=dim_mults,
        channels=channels,
        resnet_block_groups=resnet_block_groups,
        block_per_layer=block_per_layer,
    )

    model = GaussianDiffusion(
        unet,
        image_size=z_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
        p2_loss_weight_gamma=p2_weight_gamma,
    )

    trainer = Trainer(
        model,
        train_folder=train_folder,
        test_folder=test_folder,
        train_batch_size=batch_size,
        train_lr=lr,
        train_num_steps=train_num_steps,
        save_and_sample_every=save_sample_every,
        gradient_accumulate_every=gradient_accumulate_every,
        save_loss_every=save_loss_every,
        num_samples=num_samples,
        num_workers=num_workers,
        results_folder=results_folder,
        use_color_jitter=use_color_jitter,
    )

    if milestone:
        trainer.load(milestone)

    trainer.train()


if __name__ == '__main__':
    Fire(main)
