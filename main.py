import numpy as np
import time
import cv2
import mss
import time

weightsPath = "./data/yolov3.weights"
configPath = "./data/yolov3.cfg"
labelsPath = "./data/coco.names"

np.random.seed()

LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

(W, H) = (None, None)

monitor = {"top": 25, "left": 0, "width": 1080, "height": 1080}
args = {"confidence" : 0.55, "threshold" : 0.3}

with mss.mss() as sct:

	while True:

		last_time = time.time()

		frame = np.array(sct.grab(monitor))
		frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=False, crop=True)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(layer_names)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:

				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height

					frameWidth = monitor["width"]
					frameHeight = monitor["height"]

					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)
					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					classIDs.append(classID)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				
	
		print("[INFO] fps: {}".format(1 / (time.time() - last_time)))

		# frame = cv2.resize(frame,None,fx=0.7,fy=0.7)
		cv2.imshow("yolo",frame)
		cv2.waitKey(1)
