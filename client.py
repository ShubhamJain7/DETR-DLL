from ctypes import *
import sys
from collections import Counter
# define absolute path to model and image
model_path = 'C:/Users/dell/source/repos/DETR-DLL/models/DETRmodel.onnx'
image_path = 'C:/Users/dell/source/repos/DETR-DLL/test.jpg'

# define singular and plural forms of class labels 
CLASSES_SINGULAR = ['N/A', 'a person', 'a bicycle', 'a car', 'a motorcycle', 'an airplane', 'a bus', 'a train', 'a truck',
					'a boat', 'a traffic light', 'a fire hydrant', 'N/A', 'a stop sign', 'a parking meter', 'a bench', 
					'a bird', 'a cat', 'a dog', 'a horse', 'a sheep', 'a cow', 'an elephant', 'a bear', 'a zebra', 
					'a giraffe', 'N/A', 'a backpack', 'a umbrella', 'N/A', 'N/A', 'a handbag', 'a tie', 'a suitcase', 
					'a frisbee', 'a pair of skis', 'a snowboard', 'a sports ball', 'a kite', 'a baseball bat', 
					'a baseball glove', 'a skateboard', 'a surfboard', 'a tennis racket', 'a bottle', 'N/A', 'a wine glass',
					'a cup', 'a fork', 'a knife', 'a spoon', 'a bowl', 'a banana', 'an apple', 'a sandwich', 'an orange', 
					'broccoli', 'a carrot', 'a hot dog', 'a pizza', 'a donut', 'a cake', 'a chair', 'a couch', 
					'a potted plant', 'a bed', 'N/A', 'a dining table', 'N/A', 'N/A', 'a toilet', 'N/A', 'a tv', 'a laptop',
					'a mouse', 'a remote', 'a keyboard', 'a cell phone', 'a microwave', 'an oven', 'a toaster', 'a sink', 
					'a refrigerator', 'N/A', 'a book', 'a clock', 'a vase', 'a pair of scissors', 'a teddy bear', 
					'a hair drier', 'a toothbrush']

CLASSES_PLURAL = ['N/A', 'people', 'bicycles', 'cars', 'motorcycles', 'airplanes', 'buses', 'trains', 'trucks', 'boats', 
				  'traffic lights', 'fire hydrants', 'N/A', 'stop signs', 'parking meters', 'benches', 'birds', 'cats', 
				  'dogs', 'horses', 'multiple sheep', 'cows', 'elephants', 'bears', 'zebras', 'giraffes', 'N/A', 
				  'backpacks', 'umbrellas', 'N/A', 'N/A', 'handbags', 'ties', 'suitcases', 'frisbees', 'skis', 'snowboards',
				  'sports balls', 'kites', 'baseball bats', 'baseball gloves', 'skateboards', 'surfboards', 
				  'tennis rackets', 'bottles', 'N/A', 'wine glasses', 'cups', 'forks', 'knives', 'spoons', 'bowls', 
				  'bananas', 'apples', 'sandwiches', 'oranges', 'broccoli', 'carrots', 'hot dogs', 'pizzas', 'donuts', 
				  'cakes', 'chairs', 'couches', 'potted plants', 'beds', 'N/A', 'dining tables', 'N/A', 'N/A', 'toilets', 
				  'N/A', 'tvs', 'laptops', 'mice', 'remotes', 'keyboards', 'cell phones', 'microwaves', 'ovens', 'toasters',
				  'sinks', 'refrigerators', 'N/A', 'books', 'clocks', 'vases', 'scissors', 'teddy bears', 'hair driers',
    			  'toothbrushes']

# python definition of 'Detection' struct
class Detection(Structure):
	_fields_ = [("classId", c_int),
				("probability", c_float),
				("x1", c_int),
				("y1", c_int),
				("x2", c_int),
				("y2", c_int),]

# load dependant DLLs
opencv_core430 = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/opencv_core430.dll")
opencv_imgproc430 = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/opencv_imgproc430.dll")
opencv_imgcodecs430 = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/opencv_imgcodecs430.dll")
opencv_dnn430 = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/opencv_dnn430.dll")
onnx_runtime = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/onnxruntime.dll")

# load required DLL
lib = CDLL("C:/Users/dell/source/repos/DETR-DLL/Release/DETR-DLL.dll")

# define return type and arguments of 'doDetection' function
lib.doDetection.restype = c_int
lib.doDetection.argtypes = [c_wchar_p, c_char_p]

# call 'doDetection' function and get number of objects detected 
res = lib.doDetection(c_wchar_p(model_path), c_char_p(image_path.encode('utf-8')))
DetectedObjectsArray = Detection * res

# continue if objects were detected
if res!=0:
	# define return type and arguments of 'getDetections' function
	lib.getDetections.restype = c_int
	lib.getDetections.argtypes = [POINTER(DetectedObjectsArray), c_int]

	# define array to store 'Detection's and call the 'getDetections' function
	objects = DetectedObjectsArray()
	_ = lib.getDetections(objects, res)

	# get class labels and their frequency counts
	# assuming singular
	classLabels = [CLASSES_SINGULAR[d.classId] for d in objects]
	counts = Counter(classLabels)
	number_of_items = len(counts)

	output_string = "The image contains "
	for i,key in enumerate(counts.keys()):
		if counts[key] == 1:
			output_string = output_string + key
		# if there are multiple instances of the same object in the image, use plural form
		else:
			output_string = output_string + CLASSES_PLURAL[CLASSES_SINGULAR.index(key)]

		# rules for listing out identified objects
		if i < (number_of_items - 2):
			# add commas only if there are more tham two objects
			if number_of_items > 2:
				output_string = output_string + ", "
			else:
				output_string = output_string + " "
		# use and if its the second last object
		elif i == (number_of_items - 2):
			output_string = output_string + " and "
		# end sentence with fullstop
		else:
			output_string = output_string + "."

	print(output_string)
else:
	print("Cannot identify any objects in the image.")