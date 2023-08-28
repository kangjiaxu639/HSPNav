# HSPNav: Hierarchical Scene Prior Learning for Visual Semantic Navigation Towards Real Settings

Jiaxu Kang, Bolei Chen, Ping Zhong, Haonan Yang, and Jianxin Wang

Central South University

## Overview

 ![](C:\Users\123\Pictures\论文1\fig2.png)

Visual Semantic Navigation (VSN) aims at navigating a robot to a given target object in a previously unseen scene. To tackle this task, the robot must learn a nimble navigation policy by utilizing spatial patterns and semantic co-occurrence relations among objects in the scenario. Prevailing approaches extract scene priors from the instant visual observations and solidify them in neural episodic memory to achieve flexible navigation. However, due to the oblivion and underuse of the scene priors, these methods suffer from repeated exploration, sparse effective knowledge, and wrong decisions. To alleviate these issues, we propose a novel VSN policy, HSPNav, based on Hierarchical Scene Priors (HSP) and Deep Reinforcement Learning (DRL). The HSP contains two components, i.e., the egocentric semantic map-based Local Scene Priors (LSP) and the object relation graph-based Global Scene Priors (GSP). Then, efficient semantic navigation is achieved by employing an immediate LSP to retrieve conducive contextual memories from the GSP. By utilizing the MP3D dataset, the experimental results in the Habitat simulator demonstrate that our HSP brings a significant boost over the baselines. Furthermore, we take an essential step from simulation to reality by bridging the gap from Habitat to ROS. The migration evaluations show that HSPNav can generalize to realistic settings well and achieve promising performance.

## Installation

1. ```
   cd /your_path
   git clone git@github.com:kangjiaxu639/HSPNav.git
   cd /your_path/HSPNav
   ```

2. Download Matterport3d dataset following instruction [here](https://github.com/niessner/Matterport).

3. ```conda create -p /path/to/your/env hspnav=3.7```

4. Install [PyTorch](https://pytorch.org/):

   ```
   # CUDA 11.0
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   
   # CUDA 10.2
   pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
   ```

5. Install Habitat environment(assuming habitat 0.1.7):

   Install Habitat-sim [here](https://github.com/facebookresearch/habitat-sim).

   Install Habitat-lab [here](https://github.com/facebookresearch/habitat-lab).

   ```
   conda activate hspnav
   cd /path/to/your/habitat
   # Install habitat-sim
   git clone https://github.com/facebookresearch/habitat-sim.git
   cd /habitat-sim
   git checkout tags/v0.1.7
   pip install -r requirements.txt
   python setup.py install --headless --with-cuda
   # Install habitat-lab
   git clone https://github.com/facebookresearch/habitat-lab.git
   cd /habitat-lab
   git checkout tags/v0.1.7
   pip install -r requirements.txt
   python setup.py develop --all
   ```

6. Install dependencies:

   ```
   pip install -r requirement.txt`
   ```

## Training and Evaluating

### External knowledge for Object Relation Graph construction

The object semantic encoding is in `HSPNav/knowledge/object_encoding.json`

The relationships between different objects is in `HSPNav/knowledge/matterport3D_rels.json`

The object-object relationships encoding is in `HSPNav/knowledge/relationship_encoding.json`

### Training a Navigation Model

The command to train the HSPNav model:
```
python train_agent.py --cmplt 
    --cmplt_pretrained /local/crv/yiqing/result/train_cmplt_resized/17_cd.pth
    --conf --conf_pretrained /local/crv/yiqing/result/train_conf_4/14_fd.pth               
```
### Evaluating a Navigation Model

The evaluation task dataset is in `HSPNav/val/val.json`

The command to evaluate the provided HSPNav model with groundtruth semantic segmentation:
```
python test_agent.py --Q_pretrained /HSPNav/result/hspnav.pth --cmplt --cmplt_pretrained /HSPNav/pretrained/cmplt.pth --conf --conf_pretrained /HSPNav/pretrained/conf.pth         
```
The command to evaluate the HSPNav model with ACNet semantic segmentation output:
```
python test_agent.py --user_semantics --seg_pretrained /HSPNav/pretrained/final_seg.pth --Q_pretrained /HSPNav/result/hspnav.pth --cmplt --cmplt_pretrained /HSPNav/pretrained/cmplt_seg.pth --conf --conf_pretrained /HSPNav/pretrained/conf_seg.pth 
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgement

This work was supported in part by the National Natural Science Foundation of China under 62172443, in part by the Natural Science Foundation of Hunan Province under 2022JJ30760.
