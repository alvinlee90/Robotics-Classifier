#!/usr/bin/env python
import rospy
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import cv2

from object_classifier.srv import ClassifyObject, ClassifyObjectResponse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageClassifier():
    def __init__(self):
        # self.class_list = list()

        # Class names
        # class_range = rospy.get_param("/object_classifier/class_list/number")
        # for i in range(class_range):
        #     self.class_list.append(rospy.get_param("/object_classifier/class_list/object%d" % i))
        
        # Image dimensions
        self.img_size = rospy.get_param("/object_classifier/image/size")
        self.img_mean = rospy.get_param("/object_classifier/image/mean")
        self.img_std = rospy.get_param("/object_classifier/image/std")

        # Define the CNN model
        model_file = rospy.get_param("/object_classifier/model_path")
        input_layer = rospy.get_param("/object_classifier/layer/input")
        output_layer = rospy.get_param("/object_classifier/layer/output")

        # Load graph
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as file:
            graph_def.ParseFromString(file.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        # Get operation for input (image) and output (probabilities) of the graph
        self.input_op = graph.get_operation_by_name(input_layer)
        self.output_op = graph.get_operation_by_name(output_layer)

        self._session = tf.InteractiveSession(graph=graph)
   
        # CV Bridge
        self._cv_bridge = CvBridge()
        
        # Service
        self.classify_srv = rospy.Service('/object_classifier', ClassifyObject, self.classify_image) 

        # Publisher 
        self.image_pub = rospy.Publisher('/computer_vision/object_classifer', Image, queue_size=1)

        rospy.loginfo("Initialised object classifier")


    def classify_image(self, req):
        # Show cropped image
        self.image_pub.publish(req.image)

        # Process the image
        cv_image = self._cv_bridge.imgmsg_to_cv2(req.image, "rgb8")
        cv_image = cv2.resize(cv_image, (self.img_size, self.img_size), 
            interpolation=cv2.INTER_CUBIC)

        # Normalize image for MobileNet (subtract mean and divide by std)
        image = np.reshape(cv_image, (1, self.img_size, self.img_size, 3))
        image = np.subtract(image, self.img_mean)
        image = np.divide(image, self.img_std)

        # Feed to neural network
        results = self._session.run(self.output_op.outputs[0], 
            feed_dict={self.input_op.outputs[0]: image})
        results = np.squeeze(results)

        # classification = self.class_list[np.argmax(results)]
        
        # Classify image (maximum probability)
        ret = ClassifyObjectResponse()
        ret.object = np.argmax(results)
        ret.prob = np.amax(results)
        rospy.loginfo('[Classifier] Prediction: %d with %5f probability' % (ret.object, ret.prob))
        return ret


if __name__ == '__main__':
    try:
        rospy.init_node('image_classifier')
    
        imageClassifier = ImageClassifier()
    
        rospy.spin()    
    except rospy.ROSInterruptException:
        pass
