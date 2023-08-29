#include <octomap_generator/octomap_generator.h>
#include <semantics_point_type/semantics_point_type.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <cmath>
#include <sstream>
#include <cstring> // For std::memcpy
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include<typeinfo>
#include<cstdio>
#include<vector>
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"

using namespace cv;
Mat global_map(480, 640, CV_8UC3,Scalar(255, 255, 255));
int count = 0;
std::vector<std::vector<float>> cloud_z(480, std::vector<float>(640,-1.0));
// SemanticsOcTreeNodeBayesian* node = NULL;
//列表初始化
// octomap_ 指一个八叉树地图
template<class CLOUD, class OCTREE>
OctomapGenerator<CLOUD, OCTREE>::OctomapGenerator(): octomap_(0.05), max_range_(1.), raycast_range_(1.), global_map_(global_map){}

template<class CLOUD, class OCTREE>
OctomapGenerator<CLOUD, OCTREE>::~OctomapGenerator(){}

template<class CLOUD, class OCTREE>
void OctomapGenerator<CLOUD, OCTREE>::setUseSemanticColor(bool use)
{
  octomap_.setUseSemanticColor(use);
}

template<>
void OctomapGenerator<PCLColor, ColorOcTree>::setUseSemanticColor(bool use){}

template<class CLOUD, class OCTREE>
bool OctomapGenerator<CLOUD, OCTREE>::isUseSemanticColor()
{
  return octomap_.isUseSemanticColor();
}

template<>
bool OctomapGenerator<PCLColor, ColorOcTree>::isUseSemanticColor(){return false;}

/**
 * 模板类 向八叉树地图中插入点云
 * CLOUD PCLSemanticsBayesian,
 * OCTREE SemanticsOctreeBayesian
 */
template<class CLOUD, class OCTREE>
void OctomapGenerator<CLOUD, OCTREE>::insertPointCloud(const pcl::PCLPointCloud2::Ptr& cloud, const Eigen::Matrix4f& sensorToWorld)
{
  // Voxel filter to down sample the point cloud
  // Create the filtering object
  pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2 ());
  // Perform voxel filter
  float voxel_flt_size = octomap_.getResolution();
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (voxel_flt_size, voxel_flt_size, voxel_flt_size);
  sor.filter (*cloud_filtered);
  // Convert to PCL pointcloud
  CLOUD pcl_cloud;
  pcl::fromPCLPointCloud2(*cloud_filtered, pcl_cloud);
  //std::cout << "Voxel filtered cloud size: "<< pcl_cloud.size() << std::endl;
  // Transform coordinate
  pcl::transformPointCloud(pcl_cloud, pcl_cloud, sensorToWorld);
/***********************************************************************/
  // Eigen::Affine3f trans = Eigen::Affine3f::Identity(); // 3*3 float martix
  // trans.rotate(Eigen::AngleAxisf(-1.570795,Eigen::Vector3f(1,0,0)));
  // pcl::transformPointCloud(pcl_cloud,pcl_cloud,trans);
  // trans.rotate(Eigen::AngleAxisf(1.570795,Eigen::Vector3f(0,1,0)));
  // pcl::transformPointCloud(pcl_cloud,pcl_cloud,trans);
  // trans.rotate(Eigen::AngleAxisf(-1.570795,Eigen::Vector3f(0,0,1)));  
  // pcl::transformPointCloud(pcl_cloud,pcl_cloud,trans);
/************************************************************************/
  //tf::Vector3 originTf = sensorToWorldTf.getOrigin();
  //octomap::point3d origin(originTf[0], originTf[1], originTf[2]);
  octomap::point3d origin(static_cast<float>(sensorToWorld(0,3)),static_cast<float>(sensorToWorld(1,3)),static_cast<float>(sensorToWorld(2,3))); // 0 0 0
  octomap::Pointcloud raycast_cloud; // Point cloud to be inserted with ray casting
  int endpoint_count = 0; // total number of endpoints inserted
  for(typename CLOUD::const_iterator it = pcl_cloud.begin(); it != pcl_cloud.end(); ++it)
  {
    // Check if the point is invalid
    if (!std::isnan(it->x) && !std::isnan(it->y) && !std::isnan(it->z))
    {
      float dist = sqrt((it->x - origin.x())*(it->x - origin.x()) + (it->y - origin.y())*(it->y - origin.y()) + (it->z - origin.z())*(it->z - origin.z()));
      // Check if the point is in max_range
      if(dist <= max_range_)
      {
        // Check if the point is in the ray casting range
        if(dist <= raycast_range_) // Add to a point cloud and do ray casting later all together
        {
          raycast_cloud.push_back(it->x, it->y, it->z);
        }
        else // otherwise update the occupancy of node and transfer the point to the raycast range
        {
          octomap::point3d direction = (octomap::point3d(it->x, it->y, it->z) - origin).normalized ();
          octomap::point3d new_end = origin + direction * (raycast_range_ + octomap_.getResolution()*2);
          raycast_cloud.push_back(new_end);
          octomap_.updateNode(it->x, it->y, it->z, true, false); // use lazy_eval, run updateInnerOccupancy() when done
        }
        endpoint_count++;
      }
    }
  }
  // Do ray casting for points in raycast_range_
  if(raycast_cloud.size() > 0){
     octomap_.insertPointCloud(raycast_cloud, origin, raycast_range_, false, true);  // use lazy_eval, run updateInnerOccupancy() when done, use discretize to downsample cloud
  }
  // Update colors and semantics, differs between templates  line158
  updateColorAndSemantics(&pcl_cloud);
  // updates inner node occupancy and colors
  if(endpoint_count > 0)
    octomap_.updateInnerOccupancy();
}

template<>
void OctomapGenerator<PCLColor, ColorOcTree>::updateColorAndSemantics(PCLColor* pcl_cloud)
{
  for(PCLColor::const_iterator it = pcl_cloud->begin(); it < pcl_cloud->end(); it++)
  {
    if (!std::isnan(it->x) && !std::isnan(it->y) && !std::isnan(it->z))
    {
      octomap_.averageNodeColor(it->x, it->y, it->z, it->r, it->g, it->b);
    }
  }
  octomap::ColorOcTreeNode* node = octomap_.search(pcl_cloud->begin()->x, pcl_cloud->begin()->y, pcl_cloud->begin()->z);
  //std::cout << "Example octree node: " << std::endl;
  //std::cout << "Color: " << node->getColor()<< std::endl;
}

template<>
void OctomapGenerator<PCLSemanticsMax, SemanticsOctreeMax>::updateColorAndSemantics(PCLSemanticsMax* pcl_cloud)
{
  for(PCLSemanticsMax::const_iterator it = pcl_cloud->begin(); it < pcl_cloud->end(); it++)
  {
    if (!std::isnan(it->x) && !std::isnan(it->y) && !std::isnan(it->z))
    {
      octomap_.averageNodeColor(it->x, it->y, it->z, it->r, it->g, it->b);
        // Get semantics
        octomap::SemanticsMax sem;
        uint32_t rgb;
        std::memcpy(&rgb, &it->semantic_color, sizeof(uint32_t)); //这个semantic_color是semantic_bayesian中融合的语义色彩。
        sem.semantic_color.r = (rgb >> 16) & 0x0000ff;
        sem.semantic_color.g = (rgb >> 8)  & 0x0000ff;
        sem.semantic_color.b = (rgb)       & 0x0000ff;
        sem.confidence = it->confidence;
        octomap_.updateNodeSemantics(it->x, it->y, it->z, sem);
    }
  }
    SemanticsOcTreeNodeMax* node = octomap_.search(pcl_cloud->begin()->x, pcl_cloud->begin()->y, pcl_cloud->begin()->z);
    //std::cout << "Example octree node: " << std::endl;
    //std::cout << "Color: " << node->getColor()<< std::endl;
    //std::cout << "Semantics: " << node->getSemantics() << std::endl;
}

template<>
void OctomapGenerator<PCLSemanticsBayesian, SemanticsOctreeBayesian>::updateColorAndSemantics(PCLSemanticsBayesian* pcl_cloud)
{
  count++;//记录插入了多少次点云
  float x_max = 5.0;
  float x_min = -15.0;
  float y_max = 10.0;
  float y_min = -8.0;
  float z_max = 0;
  float z_min = 5000;
  float l; //单个像素代表的实际长度
  float a = (x_max - x_min) / 640;   //分辨率，根据实际需要设置，这里采用648*480
  float b = (y_max - y_min) / 480;
  l = a > b ? a : b;

  float local_x_min = -3.0 + current_x;
  float local_x_max = 3.0 + current_x;
  float local_y_min = -3.0 + current_y;
  float local_y_max = 3.0 + current_y;
  float  local_x = (local_x_max - local_x_min) / 128; //space_x
  float local_y = (local_y_max - local_y_min) / 128; //spcae_y
  float local_l = local_x > local_y ? local_x : local_y;
  
  std::vector<std::vector<float>> local_cloud_z(128, std::vector<float>(128,-1.0));
  Mat local_map(128, 128, CV_8UC3,Scalar(255, 255, 255));
  for(PCLSemanticsBayesian::const_iterator it = pcl_cloud->begin(); it < pcl_cloud->end(); it++)
  {
    if (!std::isnan(it->x) && !std::isnan(it->y) && !std::isnan(it->z))
    {
      octomap_.averageNodeColor(it->x, it->y, it->z, it->r, it->g, it->b); //与前面的颜色做平均
      // Get semantics
      octomap::SemanticsBayesian sem; //Bayesian 融合
      for(int i = 0; i < 3; i++)
      {
        uint32_t rgb;
        std::memcpy(&rgb, &it->data_sem[i], sizeof(uint32_t));
        sem.data[i].color.r = (rgb >> 16) & 0x0000ff;
        sem.data[i].color.g = (rgb >> 8)  & 0x0000ff;
        sem.data[i].color.b = (rgb)       & 0x0000ff;
        sem.data[i].confidence = it->data_conf[i];
      }
      SemanticsOcTreeNodeBayesian* node = octomap_.updateNodeSemantics(it->x, it->y, it->z, sem);
      
      //计算点在全局语义地图和局部语义地图对应的像素坐标
      int x = (it->x - x_min) / l;
      int y = (it->y - y_min) / l;
      int local_x = (it->x - local_x_min) / local_l;
      int local_y = (it->y - local_y_min) / local_l; 
      y = 480 - y; //作一个上下翻l
      local_y = 128 - local_y;
      
      //将颜色信息赋予全局语义地图
      // if (x > 0 && x < 640 && y>0 && y < 480 && node != 0 && cloud_z[y][x] <= it -> z)
      // {
      //   global_map.at<Vec3b>(y, x)[0] = node->getSemantics().getSemanticColor().b;
      //   global_map.at<Vec3b>(y, x)[1] = node->getSemantics().getSemanticColor().g;
      //   global_map.at<Vec3b>(y, x)[2] = node->getSemantics().getSemanticColor().r;
      //   cloud_z[y][x] == it->z;
      // }
      if (x > 0 && x < 640 && y>0 && y < 480 && node != 0 && cloud_z[y][x] <= it -> z)
      {
        global_map.at<Vec3b>(y, x)[0] = sem.data[0].color. b;
        global_map.at<Vec3b>(y, x)[1] = sem.data[0].color.g;
        global_map.at<Vec3b>(y, x)[2] = sem.data[0].color.r;
        cloud_z[y][x] == it->z;
      }
      
       //将颜色信息赋予局部语义地图
      // if (local_x > 0 && local_x < 128 && local_y>0 && local_y < 128 && node != 0 && local_cloud_z[local_y][local_x] <= it -> z)
      // {
      //   local_map.at<Vec3b>( local_y,  local_x)[0] = node->getSemantics().getSemanticColor().b;
      //   local_map.at<Vec3b>( local_y, local_x)[1] = node->getSemantics().getSemanticColor().g;
      //   local_map.at<Vec3b>( local_y,  local_x)[2] = node->getSemantics().getSemanticColor().r;
      //   local_cloud_z[ local_y][ local_x] == it->z;
      // }
      if (local_x > 0 && local_x < 128 && local_y>0 && local_y < 128 && node != 0 && local_cloud_z[local_y][local_x] <= it -> z)
      {
        local_map.at<Vec3b>( local_y,  local_x)[0] = sem.data[0].color.b;
        local_map.at<Vec3b>( local_y, local_x)[1] = sem.data[0].color.g;
        local_map.at<Vec3b>( local_y,  local_x)[2] = sem.data[0].color. r;
        local_cloud_z[ local_y][ local_x] == it->z;
      }
    }
  }
  global_map_ = global_map;
  OctomapGeneratorBase::global_map_ = global_map; //定义一个public变量, 发布全局语义地图

  transpose(local_map, local_map); //翻转语义地图
  flip(local_map, local_map, 0);  //rotate 90 
  local_map_ = local_map;
  OctomapGeneratorBase::local_map_ = local_map;//定义一个public变量, 发布局部语义地图
  
}

template<class CLOUD, class OCTREE>
bool OctomapGenerator<CLOUD, OCTREE>::save(const char* filename) 
{
  std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
  if (outfile.is_open()){
    std::cout << "Writing octomap to " << filename << std::endl;
    octomap_.write(outfile);
    outfile.close();
    std::cout << "Color tree written " << filename << std::endl;
    return true;
  }
  else {
    std::cout << "Could not open " << filename  << " for writing" << std::endl;
    return false;
  }
}

template<class CLOUD, class OCTREE>
bool OctomapGenerator<CLOUD, OCTREE>::read(const char* filename) 
{
  std::ifstream infile(filename, std::ios_base::in |std::ios_base::binary);
  if (!infile.is_open()) 
  {
    std::cout << "file "<< filename << " could not be opened for reading.\n";
    return false; 
  }
  std::cout<<"Open"<<filename<<"successful!"<<std::endl;
  //octomap_.read(infile);
  octomap_.readData(infile);
  infile.close();
  std::cout << "Read map successful! " << std::endl;
  return true;
}

//Explicit template instantiation
template class OctomapGenerator<PCLColor, ColorOcTree>;
template class OctomapGenerator<PCLSemanticsMax, SemanticsOctreeMax>;
template class OctomapGenerator<PCLSemanticsBayesian, SemanticsOctreeBayesian>;

