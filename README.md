# Q-CATN

This is the pytorch implementation of our manuscript "Q-space Guided Collaborative Attention Translation Network for Flexible Diffusion-Weighted Images Synthesis".

**Dependencies**

--python  3.8

--pytorch 1.13.0

--torchvision 0.14.0

--Pillow 9.4.0

--nibabel 4.0.2

--h5py 3.8.0

--numpy 1.21.5

--scipy 1.7.3

**Usages**

**Inference**

First put the trained weight (https://drive.google.com/file/d/10XZTDnNgrGBNWU81Ke2GuucJhUh5FTh-/view?usp=sharing) in 'checkpoint' folder, and put the testing data (B0, T1, T2 and b-vector) in 'dataset' folder.
To obtain synthesized DWI data, please run:

        python ./mains/test.py

You will get a high-resolution DWI synthesis from arbitrary q-space sampling.


**Some available options:**

--dataset: Training and testing dataset.

--configs: Training and testing configuration parameters.

--checkpoint: Model training weights folder.

--utils: The necessary functions required for inference.


**Contact**

If you have any questions, please contact us (dlmu.p.l.zhu@gmail.com)

