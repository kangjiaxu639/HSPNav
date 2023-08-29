#ifndef OCTOMAP_GENERATOR_ROS_H
#define OCTOMAP_GENERATOR_ROS_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <semantics_octree/semantics_octree.h>
#include <std_srvs/Empty.h>
#include <octomap/ColorOcTree.h>
#include <octomap/AbstractOcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <string>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_generator/octomap_generator.h>
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/Point.h"
#include <nav_msgs/Odometry.h>
#include<queue>

#define COLOR_OCTREE 0
#define SEMANTICS_OCTREE_MAX 1
#define SEMANTICS_OCTREE_BAYESIAN 2

/**
 * \brief ROS wrapper for octomap generator
 * \details Adapted from [RGBD slam](http://wiki.ros.org/rgbdslam) and [octomap server](http://wiki.ros.org/octomap_server)
 * \author Xuan Zhang
 * \data Mai-July 2018
 */
struct Target{
  double x;
  double y;
};
class OctomapGeneratorNode{
  public:
    /**
     * \brief Constructor
     * \param nh The ros node handler to be used in OctomapGenerator
     */
    OctomapGeneratorNode(ros::NodeHandle& nh);
    /// Desturctor
    virtual ~OctomapGeneratorNode();
    /// Reset values to paramters from parameter server
    void reset();
    /**
     * \brief Callback to point cloud topic. Update the octomap and publish it in ROS
     * \param cloud ROS Pointcloud2 message in arbitrary frame (specified in the clouds header)
     */
    void insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    void ArouseCallback(const std_msgs::String::ConstPtr& msg);
    void StateCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void DistanceCallback(const std_msgs::Float32 msg);
    void stop_navigation(const std_msgs::String::ConstPtr& msg);
    /**
     * \brief Save octomap to a file. NOTE: Not tested
     * \param filename The output filename
     */
    sensor_msgs::Image local_map;
    sensor_msgs::Image global_map;
    bool save(const char* filename) ;
    bool read(const char* filename) ;
    float current_state_x=0.0, current_state_y=0.0, current_state_z=0.0;
    int nav_stop = 0;
    bool achieve = false;
   std::queue <Target> t;
   float target[2] = {-999999.0, -999999.0};
   float last_traveling_distance = 0.0;
   float current_traveling_distance = 0.0;
   int trap_times;

  protected:
    OctomapGeneratorBase* octomap_generator_; ///<Octomap instance pointer
    ros::ServiceServer service_;  ///<ROS service to toggle semantic color display
    ros::ServiceServer octomapFullService;
    ros::ServiceServer octomapBinaryService;
    bool toggleUseSemanticColor(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response); ///<Function to toggle whether write semantic color or rgb color as when serializing octree
    bool octomapFullSrv(octomap_msgs::GetOctomap::Request& req, octomap_msgs::GetOctomap::Response& res);
    bool octomapBinarySrv(octomap_msgs::GetOctomap::Request& req,octomap_msgs::GetOctomap::Response& res);
    ros::NodeHandle nh_; ///<ROS handler
    ros::Publisher fullmap_pub_,binarymap_pub_; ///<ROS publisher for octomap message
    ros::Publisher waypoint_pub;
    image_transport::Publisher global_map_pub_;
    image_transport::Publisher local_map_pub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>* pointcloud_sub_; ///<ROS subscriber for pointcloud message
    tf::MessageFilter<sensor_msgs::PointCloud2>* tf_pointcloud_sub_; ///<ROS tf message filter to sychronize the tf and pointcloud messages
    tf::TransformListener tf_listener_; ///<Listener for the transform between the camera and the world coordinates
    ros::Subscriber arouse_sub;
    ros::Subscriber state_sub;
    ros::Subscriber nav_stop_sub;
    ros::Subscriber traveling_distance_sub;
    std::string world_frame_id_; ///<Id of the world frame
    std::string pointcloud_topic_; ///<Topic name for subscribed pointcloud message
    float max_range_; ///<Max range for points to be inserted into octomap
    float raycast_range_; ///<Max range for points to perform raycasting to free unoccupied space
    float clamping_thres_max_; ///<Upper bound of occupancy probability for a node
    float clamping_thres_min_; ///<Lower bound of occupancy probability for a node
    float resolution_; ///<Resolution of octomap
    float occupancy_thres_; ///<Minimum occupancy probability for a node to be considered as occupied
    float prob_hit_;  ///<Hit probability of sensor
    float prob_miss_; ///<Miss probability of sensor
    int tree_type_; ///<0: color octree, 1: semantic octree using bayesian fusion, 2: semantic octree using max fusion
    octomap_msgs::Octomap map_msg_; ///<ROS octomap message
    octomap_msgs::Octomap map_msg_1;
  };

#endif//OCTOMAP_GENERATOR_ROS
