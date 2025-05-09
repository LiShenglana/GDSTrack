# GDSTrack
[Modality-Guided Dynamic Graph Fusion and Temporal Diffusion for Self-Supervised RGB-T Tracking](https://arxiv.org/abs/2505.03507)


---

![GDSTrack](./misc/pipeline-png.png)

## **Installation**

- Clone the repository locally

- Create a conda environment

```shell
conda env create -f env.yaml
conda activate GDSTrack
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
# (You can use `pipreqs ./root` to analyse the requirements of this project.)
```



## **Training**

- Prepare the training data:
  We use LMDB to store the training data. Please check `./data/parse_<DATA_NAME>` and generate lmdb datasets.
- Specify the paths in `./lib/register/paths.py`.

All of our models are trained on a single machine with two RTX3090 GPUs. For distributed training on a single node with 2 GPUs:

- MAT pre-training
```shell
python -m torch.distributed.launch --nproc_per_node=2 train.py --experiment=translate_template --train_set=common_pretrain
```
- Tracker training

Modify the `cfg.model.backbone.weights` in `./config/cfg_translation_track.py` to be the last checkpoint of the MAT pre-training.
```shell
python -m torch.distributed.launch --nproc_per_node=2 train.py --experiment=translate_track --train_set=common
```

[//]: # (<details>)

[//]: # (<summary><i>Arguments:</i></summary>)

[//]: # ()
[//]: # (- `-e` or `--experiment`:         the name of experiment -- check `./lib/register/experiments.py` to get more)

[//]: # (  information about each experiment.)

[//]: # (- `-t` or `--train_set`:          the name of train set -- check `./lib/register/dataset.py` to get more information)

[//]: # (  about each train set.)

[//]: # (- `--resume_epoch`:       resume from which epoch -- for example, `100` indicates we load `checkpoint_100.pth` and)

[//]: # (  resume training.)

[//]: # (- `--pretrain_name`:      the full name of the pre-trained model file -- for example, `checkpoint_100.pth` indicates we)

[//]: # (  load `./pretrain/checkpoint_100.pth`.)

[//]: # (- `--pretrain_lr_mult`:   pretrain_lr = pretrain_lr_mult * base_lr -- load pre-trained weights and fine tune these)

[//]: # (  parameters with `pretrain_lr`.)

[//]: # (- `--pretrain_exclude`:   the keyword of the name of pre-trained parameters that we want to discard -- for)

[//]: # (  example, `head` indicates we do not load the pre-trained weights whose name contains `head`.)

[//]: # (- `--gpu_id`:             CUDA_VISIBLE_DEVICES)

[//]: # (- `--find_unused`:        used in DDP mode)

[//]: # ()
[//]: # (</details>)







## **Evaluation**
We have released the evaluation results on GTOT, RGB-T234, LasHeR, and VTUAV-ST in [results](https://pan.baidu.com/s/1PSKXn37tL_hjr-yG-pYk4A?pwd=u9fm).

[//]: # (<details>)

[//]: # (<summary><i>Arguments:</i></summary>)

[//]: # ()
[//]: # (- `-e` or `--experiment`:         the name of experiment -- check `./lib/register/experiments.py` to get more)

[//]: # (  information about each experiment.)

[//]: # (- `-t` or `--train_set`:          the name of train set -- check `./lib/register/dataset.py` to get more information)

[//]: # (  about each train set.)

[//]: # (- `-b` or `--benchmark`:          the name of benchmark -- check `./lib/register/benchmarks.py` to get more information)

[//]: # (  about each benchmark.)

[//]: # (- `--test_epoch`:         ckp of which epoch -- the default value is `300` indicates we load weights from the last epoch.)

[//]: # (- `--num_process`:        max processes each time, set 0 for single-process test.)

[//]: # (- `--gpu_id`:             CUDA_VISIBLE_DEVICES)

[//]: # (- `--vis`:                show tracking result.)

[//]: # ()
[//]: # (</details>)





---
## Citation

```bibtex
@misc{GDSTrack_2025_IJCAI,
      title={Modality-Guided Dynamic Graph Fusion and Temporal Diffusion for Self-Supervised RGB-T Tracking}, 
      author={Shenglan Li and Rui Yao and Yong Zhou and Hancheng Zhu and Kunyang Sun and Bing Liu and Zhiwen Shao and Jiaqi Zhao},
      year={2025},
      eprint={2505.03507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.03507}, 
}
```

## **Acknowledgments**

- Thanks for the great [MAT](https://github.com/difhnp/MAT),
[USOT](https://github.com/VISION-SJTU/USOT),
[GMMT](https://github.com/Zhangyong-Tang/GMMT).
- For data augmentation, we use [Albumentations](https://github.com/albumentations-team/albumentations).


## **License**

This work is released under the **GPL 3.0 license**. Please see the
LICENSE file for more information.



