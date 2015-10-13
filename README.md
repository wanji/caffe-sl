# caffe-sl

`caffe`-based implementation of the Deep Similarity Learning algorithm proposed the [MM 2014 paper](http://dl.acm.org/citation.cfm?id=2654948).


## Example

```!bash
# get the MNIST dataset
./data/mnist/get_data.sh
# extract images and generate triplets
./examples/mnist_sl/create_mnist.sh
# training
./examples/mnist_sl/train_lenet.sh 
```


## Notes

### 1. Input data

`caffe-sl` loads data with `TripletImageDataLayer`, which works with a triplet list file contains image paths:

```!bash
caffe-sl $ head examples/mnist_sl/data/train.tri 
040846.png      051449.png      041185.png
024899.png      033969.png      039096.png
000520.png      022406.png      006979.png
025207.png      020904.png      020818.png
054660.png      040836.png      023925.png
009412.png      035528.png      003730.png
033029.png      011219.png      017586.png
053240.png      033959.png      007701.png
021132.png      042217.png      015489.png
021732.png      028399.png      031010.png
...
```

The first column consists of query images, the second column consists of positive images, and the third column consists of negative images.

### 2. Loss layers

The example use cosine similarity.
The output of the last feature layer is feed to a `SliceLayer`, which split the features into three parts: query, positive, and negative features.
The two `slice_point`s shoud be set to `batch_size` and `batch_size * 2`.

The similarities between query and positive images, query and negative images will be calculated separately. 
Then, these similarities will be feed to the `PairwiseRankingLossLayer`.


## Citation

Please cite the following paper if you use this code:

```
@inproceedings{wan2014deep,
  title={Deep learning for content-based image retrieval: A comprehensive study},
  author={Wan, Ji and Wang, Dayong and Hoi, Steven Chu Hong and Wu, Pengcheng and Zhu, Jianke and Zhang, Yongdong and Li, Jintao},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  pages={157--166},
  year={2014},
  organization={ACM}
}
```


------------------------------------------------

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
