# Multi-Focus image fusion
the project is implemented with pytorch

Dataset：[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

* #### Requirements
    * pytorch 1.4.0
    * cuda 10.1

* #### Install Other Packages
   ```shell
    pip install -r requirements.txt 
    ```
* #### Generate Test Dataset
   ```shell
    python generate_test_data.py -s data/test -v data/VOCdevkit/VOC2007/JPEGImages
    ```

* #### Train

  * First edit parameters in config.json 

  * Begin to train

    ```shell
    python train.py
    ```
  * Tensorboard
    ```shell
    run tensorboard in saved/log/
    ``` 
 

* #### Evaluate 

   ```
   python evaluate.py -s data/fusion_result -l data/lytro -m saved/models/FusionCGAN/1228_224519/checkpoint-netG-epoch20.pth
   ```

- #### Model Weights：

  [download url](https://pan.baidu.com/share/init?surl=jQ9DbgPn0PdIWARtXsrAuQ)		code：xvep
 