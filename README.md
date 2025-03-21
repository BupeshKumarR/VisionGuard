# VisionGuard

This project focuses on developing a license plate detection system using YOLOv8, a state-of-the-art object detection algorithm. The system is designed to automatically identify and locate license plates in images of vehicles.

Project Overview
The project consists of several key components:

	1.	Data Preparation: The system converts XML annotations to YOLO format, which is required for training YOLOv8 models. This process extracts bounding box information for license plates from XML files and converts it to the appropriate YOLO format.
	2.	Dataset Organization: The project includes functionality to split the dataset into training and validation sets. This ensures that the model can be properly trained and evaluated on separate data subsets.
	3.	Configuration: A YAML file is automatically generated to specify dataset paths, the number of classes (in this case, one class for ‘licence’), and class names. This configuration file is crucial for the YOLO training process.
	4.	Model Training: The system utilizes the Ultralytics YOLO library to train a YOLOv8 model on the prepared dataset. The training process is configured to run for 10 epochs with specific image size and batch settings.
	5.	Validation and Inference: After training, the model is validated on the validation set to assess its performance. The system also includes functionality to perform inference on new images, demonstrating the model’s ability to detect license plates in unseen data.
 
Key Features

	•	Automated conversion of XML annotations to YOLO format
	•	Dataset splitting and organization for training and validation
	•	YOLOv8 model training with customizable parameters
	•	Model validation and inference capabilities
	•	Focused on single-class detection (license plates)
 
This project streamlines the process of creating a custom license plate detection system, from data preparation to model deployment, making it easier for researchers and developers to implement object detection for specific use cases.
