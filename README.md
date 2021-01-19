# Transformer-based Multi-Aspect Modeling for MAMS
[Transformer-based Multi-Aspect Modeling for Multi-Aspect Multi-Sentiment Analysis](https://arxiv.org/abs/2011.00476). Zhen Wu, Chengcan Ying, Xinyu Dai, Shujian Huang, Jiajun Chen. NLPCC 2020.

## Requirements
* pytorch=1.7.1
* python=3.7

## Usage
Download the pretrained RoBERTa model ([link](https://pan.baidu.com/s/1i-5qCJ57Cx46NysQdiXUWA), password:`2fv2`) and unzip it into the folder `pretrained`.

The MAMS data is preprocessed by the script `preprocess.py`. The original and preprocessed versions of the data are provided in the folder `data`. 

### For the ATSA subtask
Run the command `python main_ATSA.py` to train and test the ATSA model.

You can change training settings in the file `configs.py`.

### For the ACSA subtask
Run the command `python main_ACSA.py` to train and test the ACSA model.

You can change training settings in the file `configs.py`.

## Citation
If you use the code, please cite our paper:

```bibtex
@InProceedings{10.1007/978-3-030-60457-8_45,
author="Wu, Zhen
and Ying, Chengcan
and Dai, Xinyu
and Huang, Shujian
and Chen, Jiajun",
editor="Zhu, Xiaodan
and Zhang, Min
and Hong, Yu
and He, Ruifang",
title="Transformer-Based Multi-aspect Modeling for Multi-aspect Multi-sentiment Analysis",
booktitle="Natural Language Processing and Chinese Computing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="546--557",
isbn="978-3-030-60457-8"
}
```

## Reference
[1]. Zhen Wu, Chengcan Ying, Xinyu Dai, Shujian Huang, Jiajun Chen. [Transformer-based Multi-Aspect Modeling for Multi-Aspect Multi-Sentiment Analysis](https://arxiv.org/abs/2011.00476). NLPCC 2020.

[2]. Qingnan Jiang, Lei Chen, Ruifeng Xu, Xiang Ao, Min Yang. [A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis](https://www.aclweb.org/anthology/D19-1654.pdf). EMNLP-IJCNLP 2019.