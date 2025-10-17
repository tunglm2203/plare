import os
import io
import pickle
import argparse
import collections
import numpy as np
import multiprocessing as mp

from research.datasets.replay_buffer import storage
from research.utils import utils
from research.ai_feedback.vlm_query import KeyHolder, get_feedback_from_vlm_for_sequence
from tqdm import tqdm


def query_vlm(params):
    key_holder, image_1, image_2, model_names, env_name, rank = params
    return_list = []
    preference = get_feedback_from_vlm_for_sequence(
        key_holder=key_holder,
        image_1=image_1,
        image_2=image_2,
        model_names=model_names,
        env_name=env_name,
        rank=rank
    )
    return_list.append((preference, rank))
    return return_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True, help="mw_drawer-open-v2")
    parser.add_argument("--n_processes", type=int, default=1, help="Number of processes to query VLM.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    env_name = args.env_name.replace('mw_', '')
    if args.env_name == "mw_drawer-open-v2":
        data_path = "datasets/pref/mw_drawer-open-v2_ep2500_n0.3.npz"   # Download from CPL's repo
        image_file = "datasets/pref_image_only/mw_drawer-open-v2_ep2500_n0.3_img200.pkl"    # You need to render

    elif args.env_name == "mw_sweep-into-v2":
        data_path = "datasets/pref/mw_sweep-into-v2_ep2500_n0.3.npz"    # Download from CPL's repo
        image_file = "datasets/pref_image_only/mw_sweep-into-v2_ep2500_n0.3_img200.pkl"     # You need to render

    elif args.env_name == "mw_plate-slide-v2":
        data_path = "datasets/pref/plate-slide-v2_ep2500_n0.3.npz"      # Download from CPL's repo
        image_file = "datasets/pref_image_only/plate-slide-v2_ep2500_n0.3_img200.pkl"       # You need to render

    elif args.env_name == "mw_door-open-v2":
        data_path = "datasets/pref/mw_door-open-v2_ep2500_n0.3.npz"     # Download from CPL's repo
        image_file = "datasets/pref_image_only/mw_door-open-v2_ep2500_n0.3_img200.pkl"      # You need to render

    else:
        raise NotImplementedError

    # Load the data
    dataset = storage.load_data(data_path, exclude_keys=["mask"])
    if os.path.exists(image_file):
        with open(image_file, 'rb') as f:
            image_data = pickle.load(f)
    else:
        print(f"Cannot find image file: {image_file}, please run this file to render: scripts/render_metaworld_dataset.py")
        exit()

    # In image_data, while rendering we did down-sampling to reduce segment length from 64 to 8 for saving storage
    selected_indices = np.array([0, 4, 7])  # down-sampled segments only contains 8 images

    # Dataset size
    dataset_size = image_data['image'].shape[0]
    segment_length = 64

    model_names = ("gemini-2.0-flash", "gemini-2.0-flash")
    # Place your keys to following list
    key_list = [
        "your_gemini_key_here"
    ]
    key_holder = KeyHolder(key_list, args.n_processes)

    utils.print_green(f"Env: {env_name}")
    metrics = collections.defaultdict(list)
    n_pairs = dataset_size // 2

    start_segment_idx = len(metrics['vlm_label'])
    list_indices = list(range(start_segment_idx, n_pairs, args.n_processes))
    for i in tqdm(list_indices):
        idxs = np.arange(i, i + args.n_processes)

        segment_1_image = image_data["image"][idxs]  # (B, S, HWC)
        segment_2_image = image_data["image"][idxs + n_pairs]  # (B, S, HWC)

        image_1 = segment_1_image[:, selected_indices, :, :, :]
        image_2 = segment_2_image[:, selected_indices, :, :, :]

        if args.n_processes == 1:
            # Single process for querying
            preference = get_feedback_from_vlm_for_sequence(
                key_holder=key_holder,
                image_1=image_1[0],  # Remove batch dim
                image_2=image_2[0],  # Remove batch dim
                model_names=model_names,
                env_name=env_name,
                rank=0
            )
            vlm_label = preference
            metrics["vlm_label"].append(vlm_label)

        else:
            # Multiple processes for querying
            args_list = []
            for rank in range(args.n_processes):
                args_list.append((key_holder, image_1[rank], image_2[rank], model_names, env_name, rank))

            with mp.Pool(args.n_processes) as pool:
                outputs = pool.map(query_vlm, args_list)

            # Construct outputs from multi-processes into same original order
            # Sort all processes
            chunks_order = []
            for i in range(args.n_processes):
                chunks_order.append(outputs[i][0][1])
            sorted_chunk_indices = np.argsort(chunks_order)

            multi_vlm_labels = []
            for chunk_idx in sorted_chunk_indices:
                cur_chunk = outputs[chunk_idx]

                assert len(cur_chunk) == 1
                multi_vlm_labels.append(cur_chunk[0][0])

            vlm_label = multi_vlm_labels
            metrics["vlm_label"].extend(vlm_label)

    dataset.update(metrics)  # Update the dataset to contain the new metrics

    # Save the feedback data to a path.
    filename = os.path.basename(data_path)
    output_path = os.path.join("datasets/vlm_feedback", filename[:-4] + "_vlm_label.npz")
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **dataset)
        bs.seek(0)
        with open(output_path, "wb") as f:
            f.write(bs.read())


