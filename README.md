# MECON
Multi-Expert Coordination via Objective Negotiation for Drug-Drug Interaction Prediction

# Abstract

Accurate prediction of drug–drug interactions is crucial for clinical medication safety. In existing multimodal approaches, modality collapse and single-modality dominance are often observed during the fusion of molecular structures and knowledge graphs, primarily due to reliance on a single optimization objective. To address this limitation, MECON, a multi-expert coordination network based on objective negotiation, is proposed, in which multi-expert integration is reformulated as a multi-objective negotiation problem. The predictive contribution of each modality is adaptively balanced by dynamically computing a joint gradient update direction under the simplex constraint. Under this coordination mechanism, two domain experts are synergistically integrated. The Structure expert is designed to utilize an Inter-aware Graph Transformer, through which microscopic structural dependencies of drug pairs are explicitly captured and global chemical semantics are incorporated. The KG expert is employed to capture macroscopic far-field semantics and high-order topological relations within an extracted Adaptive Context Graph. Experimental results demonstrate that state-of-the-art baselines across different methodological paradigms are surpassed by MECON, ACC values above 0.99 are achieved on both public datasets, and microscopic and macroscopic features are effectively integrated.

# 1. Requirements

To reproduce **MECON**, the python==3.8, pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge are required.

Of course, you can create your environment by env.yaml:
```sh
    $ conda env create -f env.yaml
```

# 2. Usage

### 2.1. Dataset

Dataset for MECON can be downloaded from [DSN-DDI](https://doi.org/10.1093/bib/bbac597).

### 2.2. Useage 
For Train:
```sh
    $ python train.py
```
> **Note:** `train.py` uses the **warm-start configuration** by default.
> If you would like to test the **cold-start setting**, please replace the training, validation, and test paths with the corresponding cold-start paths, and modify the parameter configuration according to the descriptions provided in the paper.
>
> To directly reproduce the results reported in the paper, the model weights can be downloaded from the provided **[model link](https://pan.baidu.com/s/1TtEr-CSArlqpLmpAjuaf5A?pwd=0000)**. We will release all trained weights for both the **warm-start** and **cold-start** settings.
# 3. Concat
Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at niudongjiang@qdu.edu.cn.
