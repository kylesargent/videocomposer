import sys
import mediapy
from tqdm import tqdm
import os
import json


def load_various_sgm(h, w, droid_downsample, dataset):
    sys.path.insert(0, "/home/jupyter/generative-models/")
    import sgm

    import mediapy
    from sgm.data.massive import webdataset_utils
    from sgm.data.massive import webdataset_co3d
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from sgm.util import exists, instantiate_from_config, isheatmap
    import torch

    config = OmegaConf.load(
        "/home/jupyter/generative-models/configs/example_training/svd_train.yaml"
    )
    config2 = OmegaConf.load(
        "/home/jupyter/generative-models/configs/example_training/sketch.yaml"
    )
    config = OmegaConf.merge(config, config2)

    config.data.params.train_config.batch_size = 1
    config.data.params.val_config.batch_size = 1

    # config.data.params.train_config.dataset_config_1.dataset_url = "gs://xcloud-shared/kylesargent/mit_unresized/mit_val__{shard:04d}.tar"
    # config.data.params.train_config.dataset_config_1.dataset_n_shards = 1
    # config.data.params.train_config.dataset_config_1.dataset_url = "gs://xcloud-shared/kylesargent/bucketed_test/sstk__00010.tar"
    # config.data.params.train_config.dataset_config_1.dataset_n_shards = 1
    # config.data.params.train_config.dataset_config_1.dataset_url = "gs://xcloud-shared/kylesargent/flow_bucketed_sstk_576p_16frames_3fps_v2/sstk__99999.tar"
    # config.data.params.train_config.dataset_config_1.dataset_n_shards = 1

    if dataset == "re10k":
        config.data.params.val_config.dataset_config_1.dataset_url = (
            "gs://xcloud-shared/kylesargent/re10k_test.tar"
        )
    else:
        config.data.params.val_config.dataset_config_1.dataset_url = (
            "gs://xcloud-shared/kylesargent/co3d_highres_v2/co3d__00027.tar"
        )

    config.data.params.train_config.shuffle_buffer_size = 0
    config.data.params.train_config.num_workers = 0
    # config.data.params.train_config.downsample_f = "(192, 128)"
    # config.data.params.val_config.downsample_f = "(192, 128)"
    config.data.params.train_config.num_frames = 14
    config.data.params.val_config.num_frames = 14

    config.data.params.train_config.balance = 1_000
    # config.data.params.train_config.downsample_f = "(384, 256)"
    # config.data.params.val_config.downsample_f = "(384, 256)"
    config.data.params.train_config.downsample_f = f"({w}, {h})"
    config.data.params.val_config.downsample_f = f"({w}, {h})"
    config.data.params.train_config.prefetch_factor = 2
    dm = instantiate_from_config(config.data)

    from sgm.data import video_metrics

    def droid_slam_getter():
        droid_slam = video_metrics.load_droid_slam(
            w=w // droid_downsample, h=h // droid_downsample
        )
        return droid_slam

    sys.path = sys.path[1:]
    intrinsics = torch.as_tensor([h / 2, h / 2, w / 2, h / 2], device="cuda")

    import sgm

    return dm, droid_slam_getter, intrinsics, video_metrics


def main():
    h = w = 256
    n = 100

    for dataset in ["re10k", "co3d"]:
        dm, *_ = load_various_sgm(h=h, w=w, droid_downsample=1, dataset=dataset)
        dl = iter(dm.val_dataloader())
        for batch, idx in tqdm(zip(dl, range(n)), total=n):
            target_video = batch["jpg"]
            target_video = target_video.permute((0, 2, 3, 1)).cpu().numpy() / 2 + 0.5

            base_path = os.path.join(
                "/home/jupyter/videocomposer/benchmarking/",
                dataset,
            )
            video_path = os.path.join(base_path, "%.5d_target.mp4" % idx)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            mediapy.write_video(video_path, target_video)

            image_path = os.path.join(base_path, "%.5d_input.png" % idx)
            image = (
                batch["cond_frames_without_noise"]
                .permute((0, 2, 3, 1))
                .cpu()
                .numpy()[0]
                / 2
                + 0.5
            )

            data_path = os.path.join(base_path, "%.5d_data.json" % idx)
            data = {"fov": batch['aux_fov_deg'].ravel()[0].item()}
            with open(data_path, 'w') as fp:
                json.dump(data, fp)

            mediapy.write_image(image_path, image)
            # raise


if __name__ == "__main__":
    main()
