import csv
import os

import network
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset import TestDataset
from mada.config import get_hydra_config_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_performance


def run_test(
        samples_df,
        input_shape,
        batch_size,
        features_root,
        model_path,
        output_path,
        device,

):
    test_dataset = TestDataset(samples_df, input_shape=input_shape)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=len(os.sched_getaffinity(0)), pin_memory=True)

    print("Number of test images:", len(test_loader.dataset))
    model =  torch.nn.DataParallel(network.AEMAD(in_channels=3, features_root=features_root))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)
    model.eval()

    mse_criterion = torch.nn.MSELoss(reduction="none").to(device)

    test_scores, gt_labels, test_scores_dict = [], [],[]

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            raw, labels, img_ids = data["images"].to(device), data["labels"], data["img_path"]
            _, output_raw = model(raw)

            scores = mse_criterion(output_raw, raw).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(scores, axis=3), axis=2), axis=1)
            test_scores.extend(scores)

    return test_scores
    """gt_labels.extend(1 - labels.data.numpy())
            for j in range(labels.shape[0]):
                l = "attack" if labels[j].detach().numpy() == 1 else "bonafide"
                test_scores_dict.append({"img_path":img_ids[j], "labels":l, "prediction_score":float(scores[j])})

    eer, eer_th = get_performance(test_scores, gt_labels)
    print("Test EER:", eer*100)

    with open(output_path, mode="w") as csv_file:
        fieldnames = ["img_path", "labels", "prediction_score"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for d in test_scores_dict:
            writer.writerow(d)
        print("Prediction scores write done in", output_path)
    """



from pathlib import Path

import hydra
import pandas as pd

from mada.config.protocols import (
    list_preprocessed_samples,
    list_subsets_for_gathered_set,
)
from mada.utils import fix_randomness
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path=get_hydra_config_path(),
    config_name="spl_mad_testing",
)
def main(cfg: DictConfig):
    fix_randomness(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.concat(
        [
            list_preprocessed_samples(**subset).assign(**subset)
            for subset in list_subsets_for_gathered_set(cfg.test_set)
        ]
    ).query("group=='test'")

    output_path = Path(cfg.output_dir) / "test_scores.nc"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = run_test(
        samples_df=df,
        input_shape=cfg.input_shape,
        batch_size=cfg.batch_size,
        features_root=cfg.features_root,
        model_path=cfg.model_path,
        output_path=output_path,
        device=device
    )

    df['score'] = scores

    df.drop(columns='path').to_xarray().assign_attrs({'backbone': 'spl_mad'}).set_coords('universal_key').drop_vars('index').rename_dims({'index': 'sample'}).to_netcdf(output_path)






if __name__ == "__main__":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    main()

