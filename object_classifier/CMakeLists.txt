cmake_minimum_required(VERSION 2.8.3)
project(object_classifier)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  message_generation
)

add_service_files(
  FILES
  ClassifyObject.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

## Declare a catkin package
catkin_package()

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS scripts/classifier.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

