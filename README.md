### Note - I am not able to maintain this repository because of shortage of time, so if anyone who is interested in helping maintain this repository can contact me and I can add you as a maintainer, Thanks. 

# Traffic Rule Violation Detection System

This project tries to detect a car whenever it crosses a Red Light or overspeeds.
It uses tensorflow with an ssd object detection model to detect cars and from the detections in each frame each vehicle can be tracked across a video and can be checked if it crossed a redlight and speed of that vehicle can be calculated.

## Getting Started

The project is made by using tensorflow so you must be familiar with tensorflow and basic object detection and you must also know basic maths for understanding the tracking algorithm. You must be also familiar with linux OS as I have made this on Ubuntu and didn't test on other platforms.

### Prerequisites

Python packages to be installed

```
* Tensorflow (Tensorflow-gpu if you have Nvidia GPU)
* openCV
* imutils
* Pillow
* numpy
* tkinter
* urllib
* openALPR api
```
Make account on openalpr and get api secret key from [OpenALPR](https://www.openalpr.com/)

## Installing

Clone the repo and paste your secret key in [VehicleMoniter.py](https://github.com/ShreyAmbesh/Traffic-Rule-Violation-Detection-System/blob/master/VehicleMoniter.py) file on line 58.
run the project by the command ```python3 VehicleMoniter.py```


## Working Preview

![alt text](https://github.com/ShreyAmbesh/Traffic-Rule-Violation-Detection-System/blob/master/Screenshot1.png)

![alt text](https://github.com/ShreyAmbesh/Traffic-Rule-Violation-Detection-System/blob/master/Screenshot2.png)

##### Note
Do not run the file in the object detection folder cloned from tensorflow as I have made some changes to the files.

### Issues
If you find any problem you can contact me or raise an issue.

## Built With

* [Tensorflow](https://www.tensorflow.org/) - ML library
* [OpenALPR](https://www.openalpr.com/) - For detecting license plate and extracting license plate number

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/ShreyAmbesh/Traffic-Rule-Violation-Detection-System/blob/master/LICENSE) file for details
