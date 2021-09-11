# S-FPN: A shortcut feature pyramid network for sea cucumber detection in underwater images             
        
A Tensorflow implementation of S-FPN detection framework.    
You can refer to the paper [S-FPN: A shortcut feature pyramid network for sea cucumber detection in underwater images](https://www.sciencedirect.com/science/article/abs/pii/S0957417421007351)
If useful to you, please star to support my work. Thanks.    

## Configuration Environment
ubuntu(Encoding problems may occur on windows) + python3 + tensorflow1.2 + cv2 + cuda>=9.0 + GeForce GTX 1080Ti    
## Installation    
  Clone the repository    
  ```Shell    
  git clone https://github.com/FannierPeng/S-FPN_sea-cucumber.git
  ```       

## Make tfrecord   
The data is VOC format, reference [here](sample.xml)     
data path format  ($FPN_ROOT/data/io/divide_data.py)      
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│    ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```     

  ```Shell    
  cd $FPN_ROOT/data/io/  
  python convert_data_to_tfrecord.py --VOC_dir='***/VOCdevkit/VOCdevkit_train/' --save_name='train' --img_format='.jpg' --dataset='ship'
  ```

## Demo          
1、Unzip the weight $FPN_ROOT/output/res101_trained_weights/*.rar    
2、put images in $FPN_ROOT/tools/inference_image   
3、Configure parameters in $FPN_ROOT/libs/configs/cfgs.py and modify the project's root directory 
4、image slice           
  ```Shell    
  cd $FPN_ROOT/tools
  python inference.py   
  ```     
5、big image      
  ```Shell    
  cd $FPN_ROOT/tools
  python demo.py --src_folder=.\demo_src --des_folder=.\demo_des      
  ``` 

  
## Train
1、Modify $FPN_ROOT/libs/lable_name_dict/***_dict.py, corresponding to the number of categories in the configuration file    
2、download pretrain weight([resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) or [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)) from [here](https://github.com/yangxue0827/models/tree/master/slim), then extract to folder $FPN_ROOT/data/pretrained_weights    
3、    
  ```Shell    
  cd $FPN_ROOT/tools
  python train.py 
  ``` 

## Test tfrecord     
  ```Shell    
  cd $FPN_ROOT/tools    
  python $FPN_ROOT/tools/test.py  
  ``` 

## eval    
  ```Shell    
  cd $FPN_ROOT/tools   
  python eval.py
  ```  

## Summary   
  ```Shell    
  tensorboard --logdir=$FPN_ROOT/output/res101_summary/
  ```    
![01](output/res101_summary/fast_rcnn_loss.bmp) 
![02](output/res101_summary/rpn_loss.bmp) 
![03](output/res101_summary/total_loss.bmp) 

## Test results    
### sea cucumber
![11](tools/test_result/00_gt.jpg)   
![12](tools/test_result/00_fpn.jpg)       

## Note 
The method was baesd on FPN reference [FPN_Tensorflow](https://github.com/yangxue0827/FPN_Tensorflow)
    
