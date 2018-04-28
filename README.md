# KeLiPTo

----
**Ke**ras **Li**ght **P**rediction **To**ol or **KeLiPTo** aims to help deploy saved keras models (HDF5 format) only for prediction, using a small set of libraries/modules.

This will help in deploying your deep learning models, which are trained on a heavy computation capable system, on comparatively lighter systems, like SBCs(Single Board Computers, e.g. Raspberry Pi) and systems where python is not installed (C++ module only). Robotic system can take advantage from such modules where training is done offline and then model is run online to make predicitons in real-time. Another example can be deployment of prediction models on EMR clusters where you can deploy the python module using a model which is trained on EC2 deep learning AMI. Keras is used as the preferred platform as it is easier to learn.

The module development is still in progress.

For C++ module you'll need to install:

- [HDF5 C++ API](https://support.hdfgroup.org/HDF5/doc/cpplus_RM/index.html)
- [RapidJSON](https://github.com/Tencent/rapidjson)
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

Python module depends on:

- [h5py](https://www.h5py.org/)
- [json](https://docs.python.org/2/library/json.html)
- [NumPy](http://www.numpy.org/)


