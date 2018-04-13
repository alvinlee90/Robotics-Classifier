#include <iostream>
#include <cmath>
#include <math.h>
#include <ros/ros.h>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <boost/foreach.hpp>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

#include <object_classifier/ClassifyObject.h>
#include <object_detection/ObjectDetected.h>

using namespace std;

struct hsv {
    int key;
    int low_H, high_H;
    int low_S, high_S;
    int low_V, high_V;
};

class ObjectDetection
{
private:
    ros::NodeHandle n;

    ros::Subscriber depth_sub;
    
    ros::Publisher detect_result_pub_; 
    
    ros::ServiceClient classifer_client_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber color_image_sub;

    cv::Mat img_hsv_, img_rgb_;

    std::vector<hsv> colours_;

    float area_min_, area_max_; 
    float range_min_, range_max_;
    int img_width_, img_height_, img_margin_; 
    int morphology_itr;
    int depth_count_;
    float prob_threshold_;

    float x_, y_;

    pcl::PCLPointCloud2 point_cloud;
    geometry_msgs::PointStamped Camera_point;

    bool found_depth = false;
    bool found_image = false;

    tf::TransformListener* tf_listener_;
    const std::string camera_frame = "camera_depth_optical_frame";
    const std::string map_frame = "map";

public:
    ObjectDetection(const ros::NodeHandle &node):
        n(node),
        it_(n)
    {
        // Initialise shit
        IntialiseParameters(); 
        InitialiseColours(); 
        IntialiseSubscribers();
        IntialisePublishers();
        InitialiseService();
        InitialiseTransform();

        ROS_INFO("Initialised object detection");
    }
    
    void IntialiseParameters()
    {
        n.getParam("/object_detection/image/width", img_width_);
        n.getParam("/object_detection/image/height", img_height_);
        n.getParam("/object_detection/image/margin", img_margin_);

        n.getParam("/object_detection/threshold_area/min", area_min_);
        n.getParam("/object_detection/threshold_area/max", area_max_);

        n.getParam("/object_detection/range/min", range_min_);
        n.getParam("/object_detection/range/max", range_max_);

        n.getParam("/object_detection/depth/count", depth_count_);

        n.getParam("/object_detection/morphology/itr", morphology_itr);

        n.getParam("/object_detection/prob_threshold", prob_threshold_);
    }

    void InitialiseColours()
    {
        // Color parameters
        hsv red, orange, green, blue, purple, yellow;

        n.getParam("/colours/purple/key", purple.key);
        n.getParam("/colours/purple/LowH", purple.low_H);
        n.getParam("/colours/purple/HighH", purple.high_H);
        n.getParam("/colours/purple/LowS", purple.low_S);
        n.getParam("/colours/purple/HighS", purple.high_S);
        n.getParam("/colours/purple/LowV", purple.low_V);
        n.getParam("/colours/purple/HighV", purple.high_V);
        colours_.push_back(purple);

        n.getParam("/colours/red/key", red.key);
        n.getParam("/colours/red/LowH", red.low_H);
        n.getParam("/colours/red/HighH", red.high_H);
        n.getParam("/colours/red/LowS", red.low_S);
        n.getParam("/colours/red/HighS", red.high_S);
        n.getParam("/colours/red/LowV", red.low_V);
        n.getParam("/colours/red/HighV", red.high_V);
        colours_.push_back(red);

        n.getParam("/colours/orange/key", orange.key);
        n.getParam("/colours/orange/LowH", orange.low_H);
        n.getParam("/colours/orange/HighH", orange.high_H);
        n.getParam("/colours/orange/LowS", orange.low_S);
        n.getParam("/colours/orange/HighS", orange.high_S);
        n.getParam("/colours/orange/LowV", orange.low_V);
        n.getParam("/colours/orange/HighV", orange.high_V);
        colours_.push_back(orange);

        n.getParam("/colours/yellow/key", yellow.key);
        n.getParam("/colours/yellow/LowH", yellow.low_H);
        n.getParam("/colours/yellow/HighH", yellow.high_H);
        n.getParam("/colours/yellow/LowS", yellow.low_S);
        n.getParam("/colours/yellow/HighS", yellow.high_S);
        n.getParam("/colours/yellow/LowV", yellow.low_V);
        n.getParam("/colours/yellow/HighV", yellow.high_V);
        colours_.push_back(yellow);

        n.getParam("/colours/red/key", green.key);
        n.getParam("/colours/green/LowH", green.low_H);
        n.getParam("/colours/green/HighH", green.high_H);
        n.getParam("/colours/green/LowS", green.low_S);
        n.getParam("/colours/green/HighS", green.high_S);
        n.getParam("/colours/green/LowV", green.low_V);
        n.getParam("/colours/green/HighV", green.high_V);
        colours_.push_back(green);

        n.getParam("/colours/blue/key", blue.key);
        n.getParam("/colours/blue/LowH", blue.low_H);
        n.getParam("/colours/blue/HighH", blue.high_H);
        n.getParam("/colours/blue/LowS", blue.low_S);
        n.getParam("/colours/blue/HighS", blue.high_S);
        n.getParam("/colours/blue/LowV", blue.low_V);
        n.getParam("/colours/blue/HighV", blue.high_V);
        colours_.push_back(blue);
    }

    void IntialiseSubscribers()
    {
        // Subscribers 
        color_image_sub = it_.subscribe("/camera/rgb/image_color", 1, &ObjectDetection::colorimageCallBack, this);
        depth_sub = n.subscribe("/camera/depth_registered/points", 1, &ObjectDetection::depthPCCallBack, this);
    }

    void IntialisePublishers()
    {
        // Publishers 
        detect_result_pub_ = n.advertise<object_detection::ObjectDetected>("/computer_vision/result", 1);
    }

    void InitialiseService()
    {
        // Service client
        classifer_client_ = n.serviceClient<object_classifier::ClassifyObject>("/object_classifier");
    }

    void InitialiseTransform()
    {
        tf_listener_ = new tf::TransformListener; 

        bool tferr = true; 
        ROS_INFO("[ObjectDetect] Checking TF");
        tf::StampedTransform tf_map_camera; 
        while (tferr) 
        {
            tferr = false;
            try 
            {
                tf_listener_->lookupTransform(map_frame, camera_frame, ros::Time(0), tf_map_camera);
            } 
            catch(tf::TransformException &exception)
            {
                ROS_ERROR("[ObjectDetect] %s; retrying...", exception.what());
                tferr = true;
                ros::Duration(0.5).sleep(); 
                ros::spinOnce();                
            }   
        }

        ROS_INFO("[ObjectDetect] TF is working");        
    }

    virtual ~ObjectDetection() {}

    // ColorImage CallBack function
    void colorimageCallBack(const sensor_msgs::ImageConstPtr& msg)
    {
        // Convert color rosImage msg to CvImage
        try
        {
            cv_bridge::CvImagePtr img_bgr_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            
            // Convert to HSV and RGB
            cv::cvtColor(img_bgr_ptr_->image, img_hsv_, CV_BGR2HSV, 0);
            cv::cvtColor(img_bgr_ptr_->image, img_rgb_, CV_BGR2RGB, 0);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            found_image = false;
            return;
        }

        found_image = true;
    }

    // DepthImage CallBack function
    void depthPCCallBack(const boost::shared_ptr<const sensor_msgs::PointCloud2>& cloud_msg)
    {
        pcl_conversions::toPCL(*cloud_msg, point_cloud);

        found_depth = true;
        // ROS_INFO("Got depth from camera");
    }

    inline void reset_control() 
    {
        found_image = false;
        found_depth = false;
    }

    void main()
    {
        if (!found_image || !found_depth)
        {   
            reset_control();
            return;    
        }

        //Convert from BGR to HSV
        for (std::vector<hsv>::iterator it = colours_.begin(); it != colours_.end(); ++it)
        {
            cv::Mat img_threshold;
            vector<cv::Vec4i> hierarchy;
            vector<vector<cv::Point>> contours;

            // Detecting object through color
            cv::inRange(img_hsv_,
                        cv::Scalar(it->low_H, it->low_S, it->low_V),
                        cv::Scalar(it->high_H, it->high_S, it->high_V),
                        img_threshold);

            // Image morphology
            cv::Mat erode = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::Mat dilate = getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
            cv::erode(img_threshold, img_threshold, erode, cv::Point(-1,-1), morphology_itr);
            cv::dilate(img_threshold, img_threshold, dilate, cv::Point(-1,-1), morphology_itr);

            // Find contours corresponding to blocks
            cv::findContours(img_threshold, 
                             contours, 
                             hierarchy, 
                             CV_RETR_EXTERNAL, 
                             CV_CHAIN_APPROX_SIMPLE);
            
            if (hierarchy.size() > 0)
            {
                // Condition where a block is found
                // Calculate area of the block
                cv::Moments moment = cv::moments(img_threshold);
                double moment_area = moment.m00;

                if (moment_area > area_min_ && moment_area < area_max_)
                {                   
                    ROS_INFO("[ObjectDetect] Detected colour %d", it->key);
                    
                    // Case where an object is detected
                    // Coordinates of the center of the block
                    int img_u = moment.m10 / moment_area;
                    int img_v = moment.m01 / moment_area;

                    // Check depth
                    if (depth_coordinates(img_u, img_v))
                    {
                        int shape = classify_object(moment_area, img_u, img_v);
                        
                        if (shape != -1)
                        {
                            // Publish message to computer vision node
                            object_detection::ObjectDetected msg; 

                            // Object information
                            msg.colour = it->key; 
                            msg.shape = shape; 
                            
                            // Image frame [RGB]
                            cv_bridge::CvImage img_bridge = cv_bridge::CvImage(std_msgs::Header(),
                                                                               sensor_msgs::image_encodings::RGB8, 
                                                                               img_rgb_);
                            img_bridge.toImageMsg(msg.image); 
                            
                            // Object coordinates
                            msg.object_location.transform.translation.x = x_;
                            msg.object_location.transform.translation.y = y_;
                            msg.object_location.transform.translation.z = 0;

                            detect_result_pub_.publish(msg); 
                        }

                        // Send colour to the end of the vector 
                        std::rotate(colours_.begin(), it + 1, colours_.end());
                        break;
                    }
                }
            }
        }

        // Process next frame
        reset_control();
    }

    // Get World Coordinates information of the object relative to camera frame.
    bool depth_coordinates(int img_u, int img_v)
    {
        int pcl_index = ((img_v * img_height_) + img_u);
        ROS_INFO("[ObjectDetect] Checking depth at depth %d", pcl_index);

        // Container for PointCloud in XYZ type
        pcl::PointCloud<pcl::PointXYZ> points_pcl;
        pcl::fromPCLPointCloud2(point_cloud, points_pcl);
        ROS_INFO("[ObjectDetect] Checking pointcloud");

        // Get valid point from the point cloud to infer pose 
        int count = 0;
        float u, v, d;
        do
        {
            u = points_pcl.points[pcl_index].x;
            v = points_pcl.points[pcl_index].y;
            d = points_pcl.points[pcl_index].z;
            
            if (count > depth_count_)
            {
                ROS_INFO("[ObjectDetect] Over depth count");
                return false;
            }

            pcl_index++;
            count++;
        } while (isnan(u) || isnan(v) || isnan(d));

        ROS_INFO("[ObjectDetect] Depth from point cloud %f", d);

        // Classify objects within range
        if (d < range_min_ || d > range_max_)
            return false; 

        // Transform point 
        geometry_msgs::PointStamped obj, obj_tf;
        obj.point.x = u;
        obj.point.y = v;
        obj.point.z = d;
        obj.header.frame_id = camera_frame;

        try 
        {
            tf_listener_->transformPoint(map_frame, obj, obj_tf);
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR("[ObjectDetect] Fucked up TF; %s", ex.what());
            return false;
        }

        x_ = obj_tf.point.x;
        y_ = obj_tf.point.y;
        ROS_INFO("[ObjectDetect] Corrodinates are (%f, %f)", x_, y_);

        return true; 
    }

    int classify_object(double area, int img_u, int img_v)
    {      
        // Find coordinates around object
        int side = std::min((int)std::sqrt(area + img_margin_), img_height_);
        int x = std::min(std::max(img_u - side / 2, 0), img_width_ - side);
        int y = std::min(std::max(img_v - side / 2, 0), img_height_ - side);

        // Crop object
        cv::Rect square(x, y, side, side);
        cv::Mat img_crop = img_rgb_(square);   

        // Send to classifier service 
        object_classifier::ClassifyObject srv; 

        cv_bridge::CvImage img_bridge = cv_bridge::CvImage(std_msgs::Header(),
                                                           sensor_msgs::image_encodings::RGB8, 
                                                           img_crop);
        img_bridge.toImageMsg(srv.request.image); 

        // Response from classifier
        if (classifer_client_.call(srv))
        {
            if (srv.response.prob > prob_threshold_)
                return srv.response.object;
            else 
                return -1;
        }
        else
        {
            ROS_ERROR("Failed to call object classifer service");
            return -1; 
        }
    }
};

// Function main
int main(int argc, char* argv[])
{
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle nh;

    ObjectDetection object_detector(nh);
    ros::Rate loop_rate(10);

    while (ros::ok()) 
    {
        ros::Time begin = ros::Time::now();
        
        ros::spinOnce();
        object_detector.main();
        loop_rate.sleep();

        ros::Duration duration = ros::Time::now() - begin;
        //ROS_INFO("[ObjectDetect::CycleTime] Time = %f", duration.toSec());
    }

    return 0;
}
