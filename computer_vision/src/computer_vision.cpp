#include <cmath>
#include <math.h>
#include <ros/ros.h>
#include <vector>
#include <algorithm>

#include <ras_msgs/RAS_Evidence.h>
#include <std_msgs/String.h>
#include <object_detection/ObjectDetected.h>

// #include <tf/tf.h>
// #include <tf/transform_listener.h>

#include <computer_vision/CheckMap.h>

class ComputerVision
{
private:
    ros::NodeHandle n_;

    ros::ServiceClient map_check_client;

    ros::Subscriber detector_sub_;
    
    ros::Publisher evidence_pub_;
    ros::Publisher speaker_pub_;
    
    std::vector<std::vector<bool>> decision_tree_;
    std::vector<std::string> colours_;
    std::vector<std::string> shapes_; 

    std::map<int, ros::Time> object_memory_;
    float memory_; 

//    tf::TransformListener* tf_listener_;
//    const std::string camera_frame = "camera_depth_optical_frame";
   const std::string map_frame = "map";

public:
    ComputerVision(const ros::NodeHandle &node):
        n_(node)
    {
        // Initialise
        intialiseParameters(); 
        initialiseColours(); 
        initialiseObjects();       

        // Subscribers 
        detector_sub_ = n_.subscribe("/computer_vision/result", 10, &ComputerVision::detectCallback, this);

        // Publishers 
        evidence_pub_ = n_.advertise<ras_msgs::RAS_Evidence>("/evidence", 1);
        speaker_pub_ = n_.advertise<std_msgs::String>("/robot/speaker", 1); 

        // Service
        map_check_client = n_.serviceClient<computer_vision::CheckMap>("/computer_vision/map_check");

        // tf_listener_ = new tf::TransformListener; 

        // bool tferr = true; 
        // ROS_INFO("[ComputerVision] Checking TF");
        // tf::StampedTransform tf_map_camera; 
        // while (tferr) 
        // {
        //     tferr = false;
        //     try 
        //     {
        //         tf_listener_->lookupTransform(map_frame, camera_frame, ros::Time(0), tf_map_camera);
        //     } 
        //     catch(tf::TransformException &exception)
        //     {
        //         ROS_ERROR("[ComputerVision] %s; retrying...", exception.what());
        //         tferr = true;
        //         ros::Duration(0.5).sleep(); 
        //         ros::spinOnce();                
        //     }   
        // }

        // ROS_INFO("[ComputerVision] TF is working");
        ROS_INFO("[ComputerVision] Initialised computer vision");

        // Set speaker to speak the object id
        std_msgs::String speaker_msg; 
        speaker_msg.data = "Starting robot"; 
        speaker_pub_.publish(speaker_msg); 
    }
    
    void intialiseParameters()
    {
        n_.getParam("/object_detection/memory/duration", memory_);
    }

    void initialiseColours()
    {
        // Color parameters
        int key, num_colour, num_shapes;
        std::vector<bool> table; 
        
        n_.getParam("/colours/number", num_colour);        
        decision_tree_.resize(num_colour);
        colours_.resize(num_colour);

        n_.getParam("/colours/purple/key", key);
        n_.getParam("/object_classifier/decision/purple", table);
        colours_[key] = "purple_";
        decision_tree_[key] = table; 

        n_.getParam("/colours/red/key", key);
        n_.getParam("/object_classifier/decision/red", table);
        colours_[key] = "red_";
        decision_tree_[key] = table; 

        n_.getParam("/colours/orange/key", key);
        n_.getParam("/object_classifier/decision/orange", table);
        colours_[key] = "orange_";
        decision_tree_[key] = table; 

        n_.getParam("/colours/yellow/key", key);
        n_.getParam("/object_classifier/decision/yellow", table);
        colours_[key] = "yellow_";
        decision_tree_[key] = table; 

        n_.getParam("/colours/green/key", key);
        n_.getParam("/object_classifier/decision/green", table);
        colours_[key] = "green_";
        decision_tree_[key] = table; 

        n_.getParam("/colours/blue/key", key);
        n_.getParam("/object_classifier/decision/blue", table);
        colours_[key] = "blue_";
        decision_tree_[key] = table; 
    }

    void initialiseObjects() 
    {
        // Get names of the objects
        int range;
        n_.getParam("/object_classifier/class_list/number", range);

        for (int i = 0; i < range; ++i)
        {
            std::string shape; 
            std::string param = "/object_classifier/class_list/object" + std::to_string(i);
            
            n_.getParam(param, shape);
            shapes_.push_back(shape);
        }
    }

    virtual ~ComputerVision() {}

    void detectCallback(const object_detection::ObjectDetected& msg)
    {
        // Check decision tree
        if (decision_tree_[msg.colour][msg.shape])  
        {  
            int object_key = generate_key(msg.colour, msg.shape);

            // Filter repeating objects 
            if (check_memory(object_key)) 
            {
                // Transform point in camera frame to map frame
                // geometry_msgs::PointStamped obj, obj_tf;
                // obj.point.x = msg.object_location.transform.translation.x;
                // obj.point.y = msg.object_location.transform.translation.y;
                // obj.point.z = msg.object_location.transform.translation.z;
                // obj.header.frame_id = camera_frame;

                // ros::Time begin = ros::Time::now();
                // try 
                // {
                //     tf_listener_->transformPoint(map_frame, obj, obj_tf);
                // }
                // catch (tf::TransformException& ex)
                // {
                //     ROS_ERROR("[ComputerVision] Fucked up TF; %s", ex.what());
                //     return;
                // }
                // ROS_INFO("[ComputerVision] Transform time: %lf", (ros::Time::now() - begin).toSec()); 

                // Send to map_check service
                computer_vision::CheckMap srv;

                std::string object_id = colours_[msg.colour] + shapes_[msg.shape];
                if (object_id.compare("orange_star") == 0)
                    object_id = "patric";

                // srv.request.obj_map = obj_tf.point;
                srv.request.obj_map.x = msg.object_location.transform.translation.x;
                srv.request.obj_map.y = msg.object_location.transform.translation.y;
                srv.request.object_key =  object_key;
                srv.request.object_id = object_id; 

                ROS_INFO("classification: %s", object_id.c_str());
                // Response from map checking
                if (map_check_client.call(srv))
                {
                    ROS_INFO("[ComputerVision] called map_check service");
                    if (srv.response.valid)
                    {
                        ROS_INFO("[ComputerVision]valid map_check service");
                        // Publish evidence message
                        ras_msgs::RAS_Evidence evidence_msg;

                        // Header details 
                        evidence_msg.stamp = ros::Time::now();
                        evidence_msg.group_number = 2;

                        // Image frame [RGB]
                        evidence_msg.image_evidence = msg.image; 

                        // Object ID
                        evidence_msg.object_id = object_id;
                        ROS_INFO_STREAM("[ComputerVision] Detected: " << object_id);                       
                        
                        evidence_msg.object_location.transform.translation.x = srv.response.valid_obj_map.x;
                        evidence_msg.object_location.transform.translation.y = srv.response.valid_obj_map.y;
                        evidence_msg.object_location.child_frame_id = map_frame;
                        evidence_pub_.publish(evidence_msg);

                        // Set speaker to speak the object id
                        std_msgs::String speaker_msg; 
                        std::replace(object_id.begin(), object_id.end(), '_', ' '); 
                        speaker_msg.data = "Detected a " + object_id; 
                        speaker_pub_.publish(speaker_msg); 
                    }
                }
                else
                {
                    ROS_ERROR("[ComputerVision] Failed to call map_check service");
                }
            }
        }
    }

    // bool check_memory(int key)
    // {
    //     // Check memory 
    //     auto it = object_memory_.find(key); 

    //     if (it == object_memory_.end())
    //     {
    //         // Not in memory; add to memory 
    //         object_memory_.insert(std::make_pair(key, ros::Time::now() + ros::Duration(memory_)));
    //         return true; 
    //     }

    //     if (ros::Time::now() > it->second)
    //     {
    //         // Update memory
    //         object_memory_.erase(it);
    //         object_memory_.insert(std::make_pair(key, ros::Time::now() + ros::Duration(memory_)));
    //         return true;
    //     }

    //     // Already seen object in memory
    //     return false; 
    // }

    bool check_memory(int key)
    {
        // Check memory 
        // Logic: if seen twice within memory sec; send true 
        auto it = object_memory_.find(key); 

        if (it == object_memory_.end())
        {
            // Not in memory; add to memory 
            object_memory_.insert(std::make_pair(key, ros::Time::now() + ros::Duration(memory_)));
            return false; 
        }

        if (ros::Time::now() < it->second)
        {
            return true;
        }

        // Remove from memory
        object_memory_.erase(it);
        object_memory_.insert(std::make_pair(key, ros::Time::now() + ros::Duration(memory_)));
        return false;
    }

    int generate_key(int x, int y) 
    {
        // Generate key for memory map
        unsigned int key_pow = 10;
        while (y >= key_pow)
            key_pow *= 10;
        
        return x * key_pow + y;        
    }
};

// Function main
int main(int argc, char* argv[])
{
    ros::init(argc, argv, "computer_vision");
    ros::NodeHandle nh;

    ComputerVision computer_vision(nh);

    while (ros::ok()) 
    {        
        ros::spin();
    }

    return 0;
}
