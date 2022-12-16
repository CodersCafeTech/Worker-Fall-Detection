![Cover.jpg](https://usercdn.edgeimpulse.com/project158336/6345e39341793a184561b9c92b46b930d4a306d5267c16031716d0eeb7c25d07)

# Story  

In the manufacturing and construction industries, workers face serious health and safety risks every day. Workers on a job site or manufacturing floor can trip on materials and equipment or be struck by falling objects. Falls can cause serious injuries when they are not detected early. 
As a solution to this, we are developing a device that can quickly detect fall downs in the monitored area and alert to the designated person with the specific area.
The device constitutes of Raspberry Pi 4 and a camera module running with the FOMO that is capable of detecting the fall down in real-time. Each incident can be written to the database and can be displayed in the web dashboard,so the safety manager can easily check the current safety status in the monitored facility.
At the implementation level, this FOMO-based ML model can be applied to the video output from the cameras which is installed in the monitored area.

# Data aquisition and Labeling   

Data collection is the first step in every machine-learning project. Proper collection of data is one of the major factors that influence the performance of the model. It is helpful to have a wide range of perspectives and zoom levels for the items you are collecting. You may take data from any device or development board, or upload your own datasets, for data acquisition. As we have our own datasets, we upload them using the Data Acquisition tab.   
First we linked the Raspberry pi with the Edge Impulse and captured the images by attaching the camera on the roof of the building. To link the Raspberry pi with the Edge Impulse please follow this [tutorial](https://docs.edgeimpulse.com/docs/development-platforms/officially-supported-cpu-gpu-targets/raspberry-pi-4).       
The more data that neural networks have access to, the better their ability to recognize the object.   
![Data acquistion.jpg](https://usercdn.edgeimpulse.com/project158336/8d1943e203939fafeb3a0b8c3dc839f4fce919abc6e96f7eaa5d1f464462f38b)
After collecting the images we labeled it by moving onto the labeling queue. In our case we have only two labels - Standing and Fall. The surprising fact is that Edge Impulse will attempts to automate this procedure by running an object tracking algorithm in the background in order to make this labeling procedure easier.
Then we split the images between test and training set and it is very essential to validate our model. There we kept 78/22 ratio and it is better to keep the ratio like this.

# Impulse Design  

This is our Impulse. As you can see we used 96x96 images and resize mode as "Fit to shortest axes" , because in this settings FOMO performs very well. 
![Impulse.jpg](https://usercdn.edgeimpulse.com/project158336/863e234677c4730768e4bbbf676bb7b48460d42e468da712ffc6de421eac703c)
In the image tab we used Grayscale as the color depth.
Then we generated the features for our images. Even though the objects are same the features are distinguishable.
![features.jpg](https://usercdn.edgeimpulse.com/project158336/e7cab8e0cea6d64259b50f31a26d0132098917850f3b5627ea56f3593445d8e9)  

# Model       
    
Now it's time to start training the machine learning model. Generating a machine learning model from scratch requires great time and effort. Instead, we will use a technique called "transfer learning" which uses a pre-trained model on our data. That way we can create an accurate machine learning model, with fewer data inputs. Then we adjusted the training parameters to get a model with better accuracy and finally we got this.
![Neural Network settings and Architecture.jpg](https://usercdn.edgeimpulse.com/project158336/75bc2b386024088cbaf93898c13ee419863f34c1d076f35a38257163b62b05cc)       
We are using ** FOMO (MobileNet V2 0.35) ** as the neural network.     
This is our training output. We got 98% accuracy.
![training output.jpg](https://usercdn.edgeimpulse.com/project158336/0b757d5e39e6859fc48956a00edaacbd8da813e2dec3c5e7335b7fc6f08c01c2) 

By examining the confusion matrix, it is clear that the model works very well but we need to check there is a possibility of over fitting. Here is our model testing results under model testing tab and it works very well with the test data also. 

![testing_data.jpg](https://usercdn.edgeimpulse.com/project158336/3a7e0a1eec311fbb4425fbf9b907a9f74ae5f0c047f36751174ba259aea67574)      


# Testing      
For testing ,we used images which is not given in testing and training. Here we are testing 2 sample images and let's see how our model performs.      
![test1.jpg](https://usercdn.edgeimpulse.com/project158336/83618c453f9c3192f86490162e5fe32f722ab824446fca5c6e66c1e5fafcb5cf)    
![test2.jpg](https://usercdn.edgeimpulse.com/project158336/6aab28694b17aef0548e1d0e4cf8174c0cd22158b6e5ca7b945c17f0aed2f92d)  
In all our testing samples, the model performed very well, so we can go ahead and deploy it to the device.     

# Linux Python SDK   
![linux sdk.jpg](https://usercdn.edgeimpulse.com/project158336/27d30fb78fdf1aed7d4d545843d99f8a05247cad82910867639e7409aada71c1)
By using this library we can run our machine learning models on Linux machines using Python. For doing that we need to follow this [installation guide](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux/linux-python-sdk).  
Then we downloaded the model from the Edge Impulse and modified the sample code to make our project alive.      
  
# Code
The entire code and assets are provided in the [GitHub repository](https://github.com/CodersCafeTech/Worker-Fall-Detection).
