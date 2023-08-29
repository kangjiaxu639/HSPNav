#include <octomap_generator/octomap_generator_ros.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/impl/transforms.hpp>
#include <octomap_msgs/conversions.h>
#include <pcl/conversions.h>
#include <pcl/filters/passthrough.h>
#include "std_msgs/String.h"
#include <cmath>
#include <sstream>
#include <cstring> // For std::memcpy
# include <string>
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "octomap_generator/waypoint.h"
#include "geometry_msgs/Point.h"
#include <nav_msgs/Odometry.h>
#include<queue>

using namespace octomap;

float target[2] = {0.0};
//构造函数 继承ros::NodeHandle
OctomapGeneratorNode::OctomapGeneratorNode(ros::NodeHandle& nh): nh_(nh)
{
  nh_.getParam("/octomap/tree_type", tree_type_);
  // Initiate octree
  if(tree_type_ == SEMANTICS_OCTREE_BAYESIAN || tree_type_ == SEMANTICS_OCTREE_MAX)
  {
    if(tree_type_ == SEMANTICS_OCTREE_BAYESIAN)
    {
      ROS_INFO("Semantic octomap generator [bayesian fusion]");
      //pcl::PointCloud<PointXYZRGBSemanticsBayesian> -> PCLSemanticsBayesian octomap_generator.h
      //SemanticsOcTree -> SemanticsOctreeBayesian  semantics_octree.h
      octomap_generator_ = new OctomapGenerator<PCLSemanticsBayesian, SemanticsOctreeBayesian>();
    } 
    else
    {
      ROS_INFO("Semantic octomap generator [max fusion]");
      octomap_generator_ = new OctomapGenerator<PCLSemanticsMax, SemanticsOctreeMax>();
    }
    service_ = nh_.advertiseService("toggle_use_semantic_color", &OctomapGeneratorNode::toggleUseSemanticColor, this);
  }
  else
  {
    ROS_INFO("Color octomap generator");
    octomap_generator_ = new OctomapGenerator<PCLColor, ColorOcTree>();
  }
  reset();
  image_transport::ImageTransport it(nh_);
  global_map_pub_ = it.advertise("global_map", 1, true);//全局语义地图
  
  image_transport::ImageTransport it_1(nh_);
  local_map_pub_ = it_1.advertise("local_map", 1, true); //发布局部语义地图

  fullmap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_full", 1, true);

  binarymap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_binary", 1, true);

  waypoint_pub = nh_.advertise<geometry_msgs::PointStamped>("way_point",10); //发布航路点

  arouse_sub = nh_.subscribe("/arouse_fullmap", 10, &OctomapGeneratorNode::ArouseCallback,this);

  state_sub = nh_.subscribe("/state_estimation", 1, &OctomapGeneratorNode::StateCallback,this);

  nav_stop_sub = nh_.subscribe("/nav_stop",1, &OctomapGeneratorNode::stop_navigation,this);

  traveling_distance_sub = nh_.subscribe("/traveling_distance",1, &OctomapGeneratorNode::DistanceCallback,this);

  pointcloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2> (nh_, pointcloud_topic_, 5);

  tf_pointcloud_sub_ = new tf::MessageFilter<sensor_msgs::PointCloud2> (*pointcloud_sub_, tf_listener_, world_frame_id_, 5);
  tf_pointcloud_sub_->registerCallback(boost::bind(&OctomapGeneratorNode::insertCloudCallback, this, _1)); //订阅并插入点云信息
  
  octomapFullService = nh_.advertiseService("octomap_full", &OctomapGeneratorNode::octomapFullSrv, this); //服务
  octomapBinaryService = nh_.advertiseService("octomap_binary", &OctomapGeneratorNode::octomapBinarySrv, this);
}

OctomapGeneratorNode::~OctomapGeneratorNode() {}
/// Clear octomap and reset values to paramters from parameter server
void OctomapGeneratorNode::reset()
{
  nh_.getParam("/octomap/pointcloud_topic", pointcloud_topic_);
  nh_.getParam("/octomap/world_frame_id", world_frame_id_);
  nh_.getParam("/octomap/resolution", resolution_);
  nh_.getParam("/octomap/max_range", max_range_);
  nh_.getParam("/octomap/raycast_range", raycast_range_);
  nh_.getParam("/octomap/clamping_thres_min", clamping_thres_min_);
  nh_.getParam("/octomap/clamping_thres_max", clamping_thres_max_);
  nh_.getParam("/octomap/occupancy_thres", occupancy_thres_);
  nh_.getParam("/octomap/prob_hit", prob_hit_);
  nh_.getParam("/octomap/prob_miss", prob_miss_);
  nh_.getParam("/tree_type", tree_type_);
  octomap_generator_->setClampingThresMin(clamping_thres_min_);
  octomap_generator_->setClampingThresMax(clamping_thres_max_);
  octomap_generator_->setResolution(resolution_);
  octomap_generator_->setOccupancyThres(occupancy_thres_);
  octomap_generator_->setProbHit(prob_hit_);
  octomap_generator_->setProbMiss(prob_miss_);
  octomap_generator_->setRayCastRange(raycast_range_); // 投影范围
  octomap_generator_->setMaxRange(max_range_); // 最大范围
}

bool OctomapGeneratorNode::toggleUseSemanticColor(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
  octomap_generator_->setUseSemanticColor(!octomap_generator_->isUseSemanticColor());
  if(octomap_generator_->isUseSemanticColor())
    ROS_INFO("Using semantic color");
  else
    ROS_INFO("Using rgb color");
  if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
     fullmap_pub_.publish(map_msg_);
  else
     ROS_ERROR("Error serializing OctoMap");
  return true;
}

void OctomapGeneratorNode::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
  // Voxel filter to down sample the point cloud
  // Create the filtering object
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());

  pcl_conversions::toPCL(*cloud_msg, *cloud);
  // Get tf transform
  tf::StampedTransform sensorToWorldTf;
  try
  {
    tf_listener_.lookupTransform(world_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp, sensorToWorldTf);
  }
  catch(tf::TransformException& ex)
  {
    ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }
  // Transform coordinate
  Eigen::Matrix4f sensorToWorld;
  pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);
  /*********************************************/
  pcl::PassThrough<pcl::PCLPointCloud2> pass_y;
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(-0.3, 1.8);   //x高度，y调地面
  pass_y.setInputCloud(cloud);
  pass_y.filter(*cloud);               //对点云y轴方向进行过滤
  /*********************************************/
  octomap_generator_->insertPointCloud(cloud, sensorToWorld); //插入点云信息
  octomap_generator_->current_x = current_state_x;
  octomap_generator_->current_y = current_state_y; //当前机器人位置
  std_msgs::Header header = std_msgs::Header();
  header.frame_id = "map";
  local_map = *(cv_bridge::CvImage(header, "bgr8", octomap_generator_->local_map_).toImageMsg());
  local_map.encoding = "bgr8";
  
  global_map = *(cv_bridge::CvImage(header, "bgr8", octomap_generator_->global_map_).toImageMsg());
  global_map.encoding = "bgr8";

  // Publish octomap
  map_msg_.header.frame_id = world_frame_id_;
  map_msg_.header.stamp = cloud_msg->header.stamp;
  if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
     //以fullmap的形式发布整个地图
     fullmap_pub_.publish(map_msg_);
  else
     ROS_ERROR("Error serializing Full_OctoMap");

  map_msg_1.header.frame_id = world_frame_id_;
  map_msg_1.header.stamp = cloud_msg->header.stamp;
  if (octomap_msgs::binaryMapToMsg(*octomap_generator_->getOctree(), map_msg_1))
     //以binarymap的形式发布二进制地图
     binarymap_pub_.publish(map_msg_1);
  else
     ROS_ERROR("Error serializing Binary_OctoMap");
  // Publish global semantic map
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", octomap_generator_->global_map_).toImageMsg();
  global_map_pub_.publish(msg);
  // Pubshlish local semantic map
  msg = cv_bridge::CvImage(header, "bgr8", octomap_generator_->local_map_).toImageMsg();
  local_map_pub_.publish(msg);
  // 存储局部语义地图
}

//用于保存地图全部信息的服务
//由roslaunch octomap_server octomap_saver -f /home/cbl/filename.ot发出请求
bool OctomapGeneratorNode::octomapFullSrv(octomap_msgs::GetOctomap::Request  &req,
                                    octomap_msgs::GetOctomap::Response &res)
{
  ROS_INFO("Sending full map data on service request");
  
  res.map.header.frame_id = world_frame_id_;
  res.map.header.stamp = ros::Time::now();

  //save("/home/cbl/map11.ot");
  if (!octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), res.map))
    return false;

  return true;
}
//用于保存二进制地图的服务
//由roslaunch octomap_server octomap_saver /home/cbl/filename.bt发出请求
bool OctomapGeneratorNode::octomapBinarySrv(octomap_msgs::GetOctomap::Request  &req,
                                    octomap_msgs::GetOctomap::Response &res)
{
  ROS_INFO("Sending binary map data on service request");
  
  res.map.header.frame_id = world_frame_id_;
  res.map.header.stamp = ros::Time::now();

  //save("/home/cbl/map11.ot");
  if (!octomap_msgs::binaryMapToMsg(*octomap_generator_->getOctree(), res.map))
    return false;

  return true;
}
//加载.ot地图文件，发布到/octomap_full话题，测试时使用的
/*void OctomapGeneratorNode::ArouseCallback(const std_msgs::String::ConstPtr& msg)
{
  // 将接收到的消息打印出来
  ROS_INFO("I heard: [%s]", msg->data.c_str());
  octomap::OcTree* m_octree=new octomap::OcTree(0.02);
  std::string filename = msg->data;
  std::string suffix = filename.substr(filename.length()-3, 3);
  if(suffix == ".ot")
  {
      octomap::AbstractOcTree* tree = octomap::AbstractOcTree::read(filename);
      if(!tree)
      {
        ROS_ERROR("Read Full_OctoMap Failed!");
      }  
      if (m_octree){
          delete m_octree;
          m_octree = NULL;
      }
      m_octree = dynamic_cast<octomap::OcTree*>(tree);
      if (!m_octree){
        ROS_ERROR("Could not read OcTree in file, currently there are no other types supported in .ot");
      }
  }
  else
  {
    ROS_ERROR("Filename Error!");
  }
  map_msg_.header.frame_id = world_frame_id_;
  map_msg_.header.stamp = ros::Time::now();
  if (octomap_msgs::fullMapToMsg(*m_octree, map_msg_))
     fullmap_pub_.publish(map_msg_);
  else
     ROS_ERROR("Error arouse Full_OctoMap"); 
}*/
//加载.ot地图文件，发布到/octomap_full话题
void OctomapGeneratorNode::ArouseCallback(const std_msgs::String::ConstPtr& msg)
{
  // 将接收到的消息打印出来
  ROS_INFO("I heard: [%s]", msg->data.c_str());
  std::string filename = msg->data;
  /*if(!read(filename.c_str()))
  {
    ROS_ERROR("Error read Full_OctoMap!");
  }*/
  std::ifstream infile(filename.c_str(), std::ios_base::in |std::ios_base::binary);
  if (!infile.is_open()) 
  {
    std::cout << "file "<< filename << " could not be opened for reading.\n";
  }
  octomap::AbstractOcTree *tree = octomap::AbstractOcTree::read(infile);
  map_msg_.header.frame_id = world_frame_id_;
  map_msg_.header.stamp = ros::Time::now();
  if (octomap_msgs::fullMapToMsg(*tree, map_msg_))
  {
      fullmap_pub_.publish(map_msg_);
      ROS_INFO("Publish fullmap Successful!");
  }
  else
     ROS_ERROR("Error arouse Full_OctoMap"); 
}
bool OctomapGeneratorNode::save(const char* filename)
{
  octomap_generator_->save(filename);
}
bool OctomapGeneratorNode::read(const char* filename)
{
  octomap_generator_->read(filename);
}
void OctomapGeneratorNode::StateCallback(const nav_msgs::Odometry::ConstPtr& msg){
  current_state_x = msg->pose.pose.position.x;
  current_state_y = msg->pose.pose.position.y;
  current_state_z = msg->pose.pose.position.z;
  if(target[0] == -999999.0 || target[1] == -999999.0){
    target[0] = current_state_x;
    target[1] = current_state_y;
  }
}
//记录机器人每次移动的距离dis
//若dis = 0, 说明机器人陷入到陷阱中
 //当超过trap_times > 10, 生成新的导航目标
void OctomapGeneratorNode::DistanceCallback(std_msgs::Float32 msg){
  // last_traveling_distance = current_traveling_distance;
  // current_traveling_distance  = msg.data;
  // if(current_traveling_distance - last_traveling_distance != 0) ++trap_times;
  if(trap_times >= 0) ++trap_times;
}

//判断是否停止导航
void OctomapGeneratorNode::stop_navigation(const std_msgs::String::ConstPtr& msg){
  if (msg->data == "nav_stop"){
      nav_stop = 1;
      ros::shutdown();
  }
  else{
    std_msgs::Header header = std_msgs::Header();
    header.frame_id = "map";
    geometry_msgs::Point point_msg = geometry_msgs::Point();
    point_msg.x =  t.front().x;
    point_msg.y =  t.front().y;
    point_msg.z =  0.5;
    t.pop(); //出队,执行队列中最新的一个导航目标
    target[0] = point_msg.x; target[1] = point_msg.y;// target way_point坐标, 如果的当前位置不是目标物体位置,继续导航
    geometry_msgs::PointStamped way_point_msgs = geometry_msgs::PointStamped();
    way_point_msgs.header = header;
    way_point_msgs.point = point_msg;
    // trap_times = 0;
    waypoint_pub.publish(way_point_msgs); //发送way_point
    ROS_INFO("请求正常处理,响应结果:%f,%f",point_msg.x, point_msg.y);
  }
  trap_times = 0;
}

//计算两点之间的欧式距离
float cal_dis(float cur_x, float cur_y, float tar_x, float tar_y){
  float dis_x = (cur_x - tar_x) *(cur_x - tar_x);
  float dis_y = (cur_y - tar_y) *(cur_y - tar_y);
  return sqrt(dis_x + dis_y);
}
int main(int argc, char** argv)
{
  setlocale(LC_ALL,"");
  ros::init(argc, argv, "octomap_generator");  // 句柄命名空间为 octomap_generator 生成八叉树地图和2D语义地图
  ros::NodeHandle nh;
  
  /***********************************服务通信********************************************/ 
  ros::Publisher stop_check_pub = nh.advertise<std_msgs::String>("stop_check",1); //发送是否检查成功
  
  ros::ServiceClient client = nh.serviceClient<octomap_generator::waypoint>("Waypoint");
  ros::service::waitForService("Waypoint");
  OctomapGeneratorNode octomapGeneratorNode(nh);  //继承句柄nh
  int circle_count = 0;
  ros::Rate r(50);
  while(ros::ok()){
      circle_count++;
      float dis = cal_dis(octomapGeneratorNode.current_state_x, octomapGeneratorNode.current_state_y,
      octomapGeneratorNode.target[0],octomapGeneratorNode.target[1]);
      
      if(dis <= 0.2 || circle_count >= 10){
        circle_count = 0;
        // 当trap_tiems=-1时，说明到达导航点或导航点无法到达，
        //且进入封锁状态只有下一个导航点被计算出来才能重新记数； 
        //当trap_tiems=0时， 说明下一个导航点已经被计算出来
          octomapGeneratorNode.trap_times = -1; 
          octomap_generator::waypoint point;
          point.request.local_map = octomapGeneratorNode.local_map; //发送局部语义地图
          point.request.global_map = octomapGeneratorNode.global_map;

          std::string target = "table"; 
          point.request.target = target; //导航目标对象
          // stop_check infomation
          std_msgs::String stop_check;
          stop_check.data = target;
        
          bool flag = client.call(point); //发送
          if (flag)
          {
              if(point.response.x != -1.0 && point.response.y != -1.0 ){
                  //将话题发布出去
                  std_msgs::Header header = std_msgs::Header();
                  header.frame_id = "map";
                  geometry_msgs::Point point_msg = geometry_msgs::Point();
                  point_msg.x =  octomapGeneratorNode.current_state_x + point.response.x;
                  point_msg.y =  octomapGeneratorNode.current_state_y + point.response.y;
                  point_msg.z =  0.5;

                  geometry_msgs::PointStamped way_point_msgs = geometry_msgs::PointStamped();
                  way_point_msgs.header = header;
                  way_point_msgs.point = point_msg;
                  
                  Target tar;
                  tar.x = point_msg.x;
                  tar.y = point_msg.y;
                  octomapGeneratorNode.t.push(tar); //记录下一次导航的目标位置信息, 
                  stop_check_pub.publish(stop_check); //发送检查是否成功找到目标对象
                  r.sleep();
              }
          }
          else
          {
              ROS_ERROR("请求处理失败....");
              return 1;
          }
      }
       ros::spinOnce(); //开始循环，监听信息
       r.sleep();
  }
  
  return 0;
}
