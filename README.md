<div><h2>[IROS'25] Policy Learning from Large Vision-Language Model Feedback Without Reward Modeling</h2></div>
<br>

**Tung M. Luu, Donghoon Lee, Younghwan Lee, and Chang D. Yoo**
<br>
KAIST, South Korea
<br>

[[ArXiv]](https://arxiv.org/abs/2507.23391v1) [[Video]](https://youtu.be/HhhOoiVpMjU)


## Installation

To install requirements:

```
conda create --name plare python=3.9
conda activate plare
pip install -r requirements.txt --no-deps
pip install -e .
cd Metaworld
pip install -e .
```

## Setup Gemini Key:
1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. To enable parallel querying, add multiple API keys to `key_list` and adjust `n_processes` accordingly in `scripts/generate_vlm_preference.py`.

## Obtain VLM Preference:
To obtain feedback from the VLM, first render the images by running (image size of 200x200):
```
bash_scripts/script_render_visual_states.sh
```
Then, select corresponding environment in `script_generate_vlm_preference.sh` and run: 

```
bash bash_scripts/script_generate_vlm_preference.sh
```

This process takes around 10 hours. If you prefer to skip this step, you can download the pre-generated VLM preference 
labels used in the paper from [here](https://drive.google.com/drive/folders/1SwENyhHjtK1QuPjnMP-WtJ9qEIs58zpX) and 
place them in `datasets/vlm_feedback`.

## Train policies
```
bash bash_scripts/run_plare.sh
```

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@inproceedings{
    luu2025policy,
    title={Policy Learning from Large Vision-Language Model Feedback without Reward Modeling},
    author={Luu, Tung M and Lee, Donghoon and Lee, Younghwan and Yoo, Chang D},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2025}
}
```

## Acknowledgements
- This work was partially supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2021-II211381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2022-II0951, Development of Uncertainty-Aware Agents Learning by Asking Questions).

- This repo contains code adapted from [CPL](https://github.com/jhejna/cpl). We thank the authors for open-sourcing their code.

## License

MIT