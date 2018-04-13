cmake_minimum_required(VERSION 2.8.3)
project(object_detection)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS 
  object_classifier
  roscpp 
  sensor_msgs 
  std_msgs
  geometry_msgs
  cv_bridge 
  image_transport
  tf 
  pcl_conversions
  pcl_ros
  message_generation
)

find_package(OpenCV 2 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

add_message_files(
  FILES
  ObjectDetected.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  sensor_msgs 
  std_msgs
  geometry_msgs 
  cv_bridge 
  opencv2 
  image_transport
  tf
  pcl_conversions
  pcl_ros
  message_runtime
  object_classifier
)

include_directories(
	include 
	${catkin_INCLUDE_DIRS} 
	${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_executable(object_detection src/obj_det.cpp)

add_dependencies(
  object_detection
	object_classifier
	${catkin_EXPORTED_TARGETS} 
	${OpenCV_EXPORTED_TARGETS}
	${PCL_EXPORTED_TARGETS}
)

target_link_libraries(
  object_detection 
	${catkin_LIBRARIES} 
	${OpenCV_LIBRARIES}
	${PCL_COMMON_LIBRARIES} 
  ${PCL_IO_LIBRARIES})


