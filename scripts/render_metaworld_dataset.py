import gym
import numpy as np
import pickle
import argparse
from tqdm import tqdm

from research.datasets.replay_buffer import storage
from research.envs.metaworld import MetaWorldSawyerImageWrapper


def main(args):
    data = storage.load_data(args.path, exclude_keys=["mask"])
    assert "state" in data

    env = gym.make(args.env)
    env = MetaWorldSawyerImageWrapper(env, width=args.resolution, height=args.resolution)
    env.reset()  # Moves the camera

    num_segments, segment_length = data["obs"].shape[:2]
    # We do down-sampling segment length here to save storage
    all_timesteps = np.linspace(start=0, stop=segment_length - 1, num=segment_length // args.skip, endpoint=True, dtype=int).tolist()
    assert args.n_pairs * 2 <= num_segments

    # Segment 1 in a pair
    all_segments_1 = []
    for segment in tqdm(range(0, args.n_pairs), desc="Rendering for segments"):
        images = []
        for t in all_timesteps:
            env.set_state(data["state"][segment, t])
            img = env._get_image()
            images.append(img)
        all_segments_1.append(images)

    # Segment 2 in a pair
    all_segments_2 = []
    for segment in tqdm(range(0, args.n_pairs), desc="Rendering for segments"):
        images = []
        for t in all_timesteps:
            env.set_state(data["state"][segment + 10000, t])
            img = env._get_image()
            images.append(img)
        all_segments_2.append(images)

    all_segments_1 = np.array(all_segments_1, dtype=np.uint8)
    all_segments_2 = np.array(all_segments_2, dtype=np.uint8)
    all_segments = np.concatenate([all_segments_1, all_segments_2], axis=0)

    # save the data
    new_data = {"image": all_segments,}
    with open(args.output, "wb") as f:
        pickle.dump(new_data, f)
    print(f"Completed rendering for {args.env}: {args.n_pairs} segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output the new dataset")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=200, help="Resolution to render")
    parser.add_argument("--skip", type=int, default=8)
    parser.add_argument("--n_pairs", type=int, default=10000)
    args = parser.parse_args()

    main(args)
