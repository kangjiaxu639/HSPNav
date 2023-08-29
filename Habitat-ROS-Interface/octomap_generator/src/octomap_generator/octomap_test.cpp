#include <ros/ros.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include <octomap/AbstractOcTree.h>
#include <octomap_msgs/conversions.h>
using namespace std;
using namespace octomap;


int main(int argc, char** argv)
 {
  std::string filename(argv[1]);
  ros::init(argc,argv,"octomap_test");
  ros::NodeHandle nh_;
  octomap_msgs::Octomap map_msg_;
  octomap_msgs::Octomap map_msg_binary;
  ros::Publisher fullmap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_full_1", 1, true);
  ros::Publisher binarymap_pub_ = nh_.advertise<octomap_msgs::Octomap>("octomap_binary_1", 1, true);
  std::ifstream infile(filename.c_str(), std::ios_base::in |std::ios_base::binary);
  if (!infile.is_open()) 
  {
    cout << "file "<< filename << " could not be opened for reading.\n";
    return -1;
  }
  else
  {
    cout << "color tree read from "<< filename <<"\n";
  }

  AbstractOcTree *tree = AbstractOcTree::read(infile);
  infile.close();
  ColorOcTree *T = dynamic_cast<ColorOcTree*>(tree);
  map_msg_.header.frame_id = "map";
  map_msg_.header.stamp = ros::Time::now();
  if (octomap_msgs::fullMapToMsg(*tree, map_msg_))
  {
      fullmap_pub_.publish(map_msg_);
      ROS_INFO("Publish fullmap Successful!");
  }
  else
     ROS_ERROR("Error arouse Full_OctoMap"); 
  map_msg_binary.header.frame_id = "map";
  map_msg_binary.header.stamp = ros::Time::now();
  if (octomap_msgs::binaryMapToMsg(*T, map_msg_binary))
  {
      binarymap_pub_.publish(map_msg_binary);
      ROS_INFO("Publish binarymap Successful!");
  }
  else
     ROS_ERROR("Error arouse Binary_OctoMap");
  //T->write("/home/cbl/rewritemap.ot");
  //cout << "Write Successful!"<<endl;
  ros::spin();
  return 0;
}
