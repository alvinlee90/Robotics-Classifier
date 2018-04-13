cmake_minimum_required(VERSION 2.8.3)
project(computer_vision)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  sensor_msgs 
  std_msgs
  geometry_msgs
  ras_msgs 
  message_generation
)

find_package(Boost REQUIRED COMPONENTS system)

add_service_files(
  DIRECTORY 
  srv
  FILES
  CheckMap.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  sensor_msgs 
  std_msgs
  geometry_msgs 
  ras_msgs
  message_runtime
)

include_directories( include ${catkin_INCLUDE_DIRS})

add_executable(computer_vision src/computer_vision.cpp)

add_dependencies(computer_vision ${catkin_EXPORTED_TARGETS})

target_link_libraries(computer_vision ${catkin_LIBRARIES})
