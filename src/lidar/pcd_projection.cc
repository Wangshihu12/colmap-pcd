#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"
#include "pcd_projection.h"

namespace colmap{
namespace lidar{

using namespace Eigen;

// 将激光雷达点投影到图像上，并存储与图像点对应的三维点信息
void PcdProj::SetNewImage(const Image& image, const Camera& camera, std::map<point3D_t,Eigen::Matrix<double,6,1>>& map){
    // 创建新的图像结构体
    // 从图像中提取旋转和平移信息
    // 从四元数获取旋转矩阵
    Eigen::Quaterniond q_cw(image.Qvec()[0],image.Qvec()[1],image.Qvec()[2],image.Qvec()[3]);
    Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();  // 相机到世界坐标系的旋转矩阵
    Eigen::Vector3d t_cw = image.Tvec();  // 相机到世界坐标系的平移向量
    
    // 仅适用于OpenCV相机模型
    // 初始化图像参数
    std::vector<double> params = camera.Params();  // 获取相机参数
    double scale = options_.depth_image_scale;  // 深度图像缩放比例
    int img_h = static_cast<int>(camera.Height() * scale);  // 缩放后的图像高度
    int img_w = static_cast<int>(camera.Width() * scale);  // 缩放后的图像宽度

    // 保存像素位置和3D点ID
    // 遍历图像中的2D点，收集有对应3D点的特征点
    std::set<Eigen::Matrix<int,2,1>,fea_compare> features;  // 使用自定义比较器的特征点集合
    for (const Point2D& point2D : image.Points2D()){
        if (!point2D.HasPoint3D()) {  // 如果2D点没有对应的3D点，则跳过
            continue;
        } 
        Eigen::Matrix<int,2,1> uv = (point2D.XY() * scale).cast<int>();  // 缩放并转换为整数坐标
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;  // 如果点超出图像边界，则跳过
        features.insert(uv);  // 将有效点加入特征点集合
    }

    // 创建LImage对象，传入特征点集合、旋转矩阵和平移向量
    LImage img(features,rot_cw,t_cw);

    // 初始化相机参数
    img.img_height = img_h;  // 设置图像高度
    img.img_width = img_w;   // 设置图像宽度
    img.img_name = image.Name();  // 设置图像名称
    img.fx = params[0] * scale;  // 设置x方向焦距（已缩放）
    img.fy = params[1] * scale;  // 设置y方向焦距（已缩放）
    img.cx = params[2] * scale;  // 设置x方向主点（已缩放）
    img.cy = params[3] * scale;  // 设置y方向主点（已缩放）

    // 搜索地图中与当前图像对应的节点
    ImageMapType img_nodes;

    // 在子地图中搜索与当前图像视锥体相交的节点
    SearchSubMap(img, img_nodes);

    // 将激光雷达点投影到图像上
    ImageMapProj(img, img_nodes, camera);

    // 匹配图像点与激光雷达点
    for (const Point2D& point2D : image.Points2D()){
        if (!point2D.HasPoint3D()) {  // 如果2D点没有对应的3D点，则跳过
            continue;
        } 

        Eigen::Matrix<int,2,1> uv = (point2D.XY() * scale).cast<int>();  // 缩放并转换为整数坐标
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;  // 如果点超出图像边界，则跳过
        auto iter = img.feature_pts_map.find(uv);  // 在特征点映射中查找对应的雷达点
        if (iter != img.feature_pts_map.end()){  // 如果找到匹配的雷达点
            point3D_t id = point2D.GetPoint3DId();  // 获取3D点ID
            Eigen::Matrix<double,6,1> pt_lidar;  // 创建6维向量存储雷达点信息（位置+法向量）
            pt_lidar <<static_cast<double>(iter->second.first.x),  // 点的x坐标
                        static_cast<double>(iter->second.first.y),  // 点的y坐标
                        static_cast<double>(iter->second.first.z),  // 点的z坐标
                        static_cast<double>(iter->second.first.normal_x),  // 法向量x分量
                        static_cast<double>(iter->second.first.normal_y),  // 法向量y分量
                        static_cast<double>(iter->second.first.normal_z);  // 法向量z分量
            map.insert({id,pt_lidar});  // 将3D点ID和对应的雷达点信息插入映射
            img.succeed_match +=1;  // 成功匹配计数加1
        }
    }

    // 如果设置了保存深度图像选项，则保存深度图像
    if (options_.if_save_depth_image){
        SaveDepthImage(img);  // 调用函数保存深度图像
        std::cout<<"Saved depth image "<<img.img_name<<std::endl;  // 输出保存成功信息
    }
}

void PcdProj::SetNewImage(const Image& image, 
            const Camera& camera, 
            std::vector<std::pair<Eigen::Vector2d, bool>,Eigen::aligned_allocator<std::pair<Eigen::Vector2d, bool>>>& pt_xys,   // 像素坐标和匹配标志
            std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>>& pt_xyzs){           // 图像坐标对应 3D 点
    // 从图像中提取旋转和平移矩阵
    Eigen::Quaterniond q_cw(image.Qvec()[0],image.Qvec()[1],image.Qvec()[2],image.Qvec()[3]);
    Eigen::Matrix3d rot_cw = q_cw.toRotationMatrix();
    Eigen::Vector3d t_cw = image.Tvec();
    
    // Only for OpenCV camera model
    // 初始化图像参数
    std::vector<double> params = camera.Params();
    double scale = options_.depth_image_scale;
    int img_h = static_cast<int>(camera.Height() * scale);
    int img_w = static_cast<int>(camera.Width() * scale);

    std::set<Eigen::Matrix<int,2,1>,fea_compare> features;
    for (const auto& pt_xy : pt_xys){

        Eigen::Matrix<int,2,1> uv = (pt_xy.first * scale).cast<int>();
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ) continue;
        features.insert(uv);
    }

    LImage img(features,rot_cw,t_cw);
    img.img_height = img_h;
    img.img_width = img_w;
    img.img_name = image.Name();
    img.fx = params[0] * scale;
    img.fy = params[1] * scale;
    img.cx = params[2] * scale;
    img.cy = params[3] * scale;

    ImageMapType img_nodes;

    // 在子地图中搜索与当前图像视锥体相交的点云，并放入img_nodes中
    SearchSubMap(img, img_nodes);

    ImageMapProj(img, img_nodes, camera);

    // Get the equation of the plane
    double fx = params[0];
    double fy = params[1];
    double cx = params[2];
    double cy = params[3];

    for (auto& pt_xy : pt_xys){

        Eigen::Matrix<int,2,1> uv = (pt_xy.first * scale).cast<int>();
        if (uv(0)<0 || uv(0)>=img_w || uv(1)<0 || uv(1)>=img_h ){
            pt_xy.second = false;
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xyzs.push_back(pt_xyz);
            continue;
        } 

        auto iter = img.feature_pts_map.find(uv);
        if (iter != img.feature_pts_map.end()){
            pt_xy.second = true;
            // u = fx * x / z + cx
            // v = fy * y / z + cy
            // z * (u - cx)/fx = x
            // z * (v - cy)/fy = y
            // ax + by + cz + d = 0
            // (a * (u - cx)/fx + b * (v - cy)/fy + c ) * z + d = 0
            double a = static_cast<double>(iter->second.first.normal_x);
            double b = static_cast<double>(iter->second.first.normal_y);
            double c = static_cast<double>(iter->second.first.normal_z);
            double d = 0 - a * static_cast<double>(iter->second.first.x)
                         - b * static_cast<double>(iter->second.first.y)
                         - c * static_cast<double>(iter->second.first.z);
            double u = pt_xy.first(0);
            double v = pt_xy.first(1);
            double z = - d / (a * (u - cx)/fx + b * (v - cy)/fy + c );
            double x = z * (u - cx)/fx;
            double y = z * (v - cy)/fy;
            Eigen::Vector3d pt_xyz;
            pt_xyz << x,y,z;
            pt_xyzs.push_back(pt_xyz);
        } else {
            pt_xy.second = false;
            Eigen::Vector3d pt_xyz = Eigen::Vector3d::Zero();
            pt_xyzs.push_back(pt_xyz);
        }
    }    
}

// 构建子地图
void PcdProj::BuildSubMap(const MapType& ptr){
    // 设置全局地图指针，MapType 是点云地图的类型，ptr 是输入的点云地图
    global_map_ptr_ = ptr;
    // 遍历点云地图中的每个点，PointType 是点的数据类型
    for (PointType& pt : ptr->points){
        // 更新全局地图的边界坐标，记录整个点云的空间范围
        // 通过比较当前点的坐标与已知的最小/最大值，不断更新边界值
        global_map_min_x_ = std::min(global_map_min_x_, pt.x);
        global_map_max_x_ = std::max(global_map_max_x_, pt.x);
        global_map_min_y_ = std::min(global_map_min_y_, pt.y);
        global_map_max_y_ = std::max(global_map_max_y_, pt.y);
        global_map_min_z_ = std::min(global_map_min_z_, pt.z);
        global_map_max_z_ = std::max(global_map_max_z_, pt.z);

        // 获取点的索引键值，这是将点分配到特定子地图的依据
        // GetKeyType() 函数可能根据点的坐标计算网格索引或空间哈希值
        auto key = GetKeyType(pt);

        // 在子地图集合中查找是否已存在该键值对应的节点
        auto iter = submap_.find(key);

        // 如果不存在该键值的节点，创建新节点
        if (iter == submap_.end()){
            NodeType node; // 创建新的节点对象
            node.push_back(pt); // 将当前点添加到新节点
            submap_.insert({key,node}); // 将新节点插入子地图集合
            submap_num_ +=1; // 子地图数量加1
        } else {
            // 如果已存在该键值的节点，直接将点添加到现有节点
            iter->second.push_back(pt);
        }
    }
}

// 在子地图中搜索与视锥体相交的点云节点,这些节点将被添加到image_map中
void PcdProj::SearchSubMap(const LImage& img, ImageMapType& image_map){
    // 计算从相机坐标系到世界坐标系的变换
    // rot_wc: 从世界坐标系到相机坐标系的旋转矩阵的转置，即相机到世界的旋转
    Eigen::Matrix3f rot_wc = img.rot_cw.cast<float>().transpose();
    // t_wc: 相机在世界坐标系中的位置
    Eigen::Vector3f t_wc = - rot_wc * img.t_cw.cast<float>();

    // 定义相机坐标系中的基本向量
    Eigen::Vector3f center_v(0.0,0.0,1.0); // 相机光轴方向向量
    Eigen::Vector3f x_bar_v(1.0,0.0,0.0); // 相机坐标系x轴方向
    Eigen::Vector3f y_bar_v(0.0,1.0,0.0); // 相机坐标系y轴方向

    // 计算图像平面归一化坐标的边界值
    // 这些值定义了相机视锥体的范围
    float x_bar_min = -img.cx / img.fx; // 左边界归一化坐标
    float x_bar_max = (img.img_width-img.cx) / img.fx; // 右边界归一化坐标
    float y_bar_min = -img.cy / img.fy; // 上边界归一化坐标
    float y_bar_max = (img.img_height - img.cy) / img.fy; // 下边界归一化坐标

    // 计算相机坐标系中图像四个角点的方向向量（z=1平面上）
    Eigen::Vector3f corner_1 = x_bar_v * x_bar_max  + y_bar_v * y_bar_max;
    Eigen::Vector3f corner_2 = x_bar_v * x_bar_max  + y_bar_v * y_bar_min;
    Eigen::Vector3f corner_3 = x_bar_v * x_bar_min  + y_bar_v * y_bar_min;
    Eigen::Vector3f corner_4 = x_bar_v * x_bar_min  + y_bar_v * y_bar_max;
    // 将四个角点从相机坐标系转换到世界坐标系，并延伸到指定距离
    // options_.choose_meter定义了视锥体的深度
    // 公式: 世界坐标 = 相机位置 + 旋转矩阵 * (中心点 + 角点) * 深度
    corner_1 = (t_wc + rot_wc * (center_v + corner_1) * options_.choose_meter).eval();
    corner_2 = (t_wc + rot_wc * (center_v + corner_2) * options_.choose_meter).eval();
    corner_3 = (t_wc + rot_wc * (center_v + corner_3) * options_.choose_meter).eval();
    corner_4 = (t_wc + rot_wc * (center_v + corner_4) * options_.choose_meter).eval();

    // 使用相机位置和四个角点创建一个四棱锥体（视锥体）
    // 这个视锥体代表相机的可视范围
    QuadPyramid quad_pyramid(t_wc,corner_1,corner_2,corner_3,corner_4);
  
    // 在子地图中搜索与视锥体相交的点云节点
    // 这些节点将被添加到image_map中
    SearchImageMap(quad_pyramid,image_map);
}

void PcdProj::ImageMapProj(LImage& img, ImageMapType& image_map, const Camera& camera){
    // 获取相机的旋转矩阵和平移向量（世界坐标系到相机坐标系的变换）
    Eigen::Matrix3f rot_cw = img.rot_cw.cast<float>();
    Eigen::Vector3f t_cw = img.t_cw.cast<float>();

    // 如果需要保存激光雷达点云帧，则准备输出文件
    std::ofstream ofs;
    if (options_.if_save_lidar_frame) {
        // 从图像名称中提取基本名称（去掉扩展名）
        std::string substr;
        std::stringstream s_stream(img.img_name);
        std::getline(s_stream, substr, '.');

        // 构建点云输出文件路径
        std::string point_cloud_write_path = options_.lidar_frame_folder + "/" 
                                            + substr + ".txt";
        ofs.open(point_cloud_write_path, std::ios::out | std::ios::trunc);
    }

    // 获取图像相关点云节点数量
    int num = image_map.size();
    // 使用OpenMP并行处理每个点云节点
    # pragma omp parallel for
    for (int i = 0; i < num; i++){
        NodeType* node_ptr = image_map[i];
        // for (NodeType** iter = image_map.begin(); iter != image_map.end(); iter++){
        // NodeType* node_ptr = *iter;

        // 遍历节点中的每个点
        for (PointType& pt : *node_ptr){
            // 获取点的世界坐标
            Eigen::Vector3f pt_w = pt.getVector3fMap();
            // 如果需要保存激光雷达帧，则写入文件
            // 注意坐标系的转换：z, -x, -y
            if (options_.if_save_lidar_frame) {
                ofs << pt_w(2)<<" "<<-pt_w(0)<<" "<<-pt_w(1)<<std::endl;
            }

            // 将点从世界坐标系转换到相机坐标系
            Eigen::Vector3f pt_c = rot_cw * pt_w + t_cw;

            // 如果点在相机后方，跳过处理（z<0表示点在相机后方）
            if (pt_c(2) < 0) continue;
            
            // 获取相机内参
            std::vector<double> params = camera.Params();
            double fx = params[0]; 
            double fy = params[1];
            double cx = params[2];
            double cy = params[3];
            double depth_image_scale = options_.depth_image_scale; // 深度图缩放比例

            // 将点投影到图像平面上（未考虑畸变）
            // 使用针孔相机模型: u = fx * x/z + cx, v = fy * y/z + cy
            double u_ori = fx * static_cast<double>(pt_c(0) / pt_c(2)) + cx;
            double v_ori = fy * static_cast<double>(pt_c(1) / pt_c(2)) + cy;

            // 考虑相机畸变影响
            Vector2d uv_ori;
            uv_ori << u_ori, v_ori; // 未畸变的像素坐标
            Vector2d uv_dis;
            uv_dis = DistortOpenCV(uv_ori, camera); // 应用畸变模型

            // 计算缩放后的像素坐标（四舍五入）
            int u0 = int(round(uv_dis(0) * depth_image_scale));
            int v0 = int(round(uv_dis(1) * depth_image_scale));
            // lidar point near the image should have large scale
            // lidar point far from the image should have small scale
            // The most appropriate scale can make the lidar projection cover the image
            // Adjust scale by focal length

            // 根据点的深度计算投影的尺度
            // 近处的点投影区域大，远处的点投影区域小
            float dist = pt_c(2); // 点到相机的深度距离
            // scale = ax + b;
            int scale_x; // x方向投影尺度
            int scale_y; // y方向投影尺度

            // 计算与相机内参和缩放比例相关的最大和最小投影尺度
            double max_proj_scale_x = static_cast<double>(options_.max_proj_scale) * (fx/3039.0) * (depth_image_scale/0.2);
            double max_proj_scale_y = static_cast<double>(options_.max_proj_scale) * (fy/3039.0) * (depth_image_scale/0.2);

            double min_proj_scale_x = static_cast<double>(options_.min_proj_scale) * (fx/3039.0) * (depth_image_scale/0.2);
            double min_proj_scale_y = static_cast<double>(options_.min_proj_scale) * (fy/3039.0) * (depth_image_scale/0.2);

            // 计算投影尺度的线性插值参数 (scale = a * dist + b)
            static double a_x = (max_proj_scale_x - min_proj_scale_x)/
                        (options_.min_proj_dist - static_cast<double>(options_.choose_meter));
            static double b_x = min_proj_scale_x - a_x * static_cast<double>(options_.choose_meter);

            static double a_y = (max_proj_scale_y - min_proj_scale_y)/
                        (options_.min_proj_dist - static_cast<double>(options_.choose_meter));
            static double b_y = static_cast<double>(options_.min_proj_scale) - a_y * static_cast<double>(options_.choose_meter);

            // 根据点的深度确定投影尺度
            if (dist < options_.min_lidar_proj_dist) {
                continue; // 距离太近的点不投影
            } else if (options_.min_lidar_proj_dist <= dist && dist<= options_.min_proj_dist) {
                // 在近距离范围内使用最大投影尺度
                scale_x = static_cast<int>(max_proj_scale_x); 
                scale_y = static_cast<int>(max_proj_scale_y);
            } else if (dist > options_.min_proj_dist){
                // 超过最小投影距离后，尺度随距离线性减小
                scale_x = static_cast<int>(a_x * dist + b_x);
                scale_y = static_cast<int>(a_y * dist + b_y);
            } else {
                std::cout<<"Please resolve the parameter conflict"<<std::endl;
                continue;
            }

            // 在投影点周围的矩形区域内进行投影
            for (int u = u0 - scale_x; u <= u0 + scale_x; u++){
            for (int v = v0 - scale_y; v <= v0 + scale_y; v++){
                // 检查投影点是否在图像范围内
                if(u < 0 || u >= img.img_width || v < 0 || v >= img.img_height){
                    continue;
                }

                // 创建像素坐标
                Eigen::Matrix<int,2,1> uv;
                uv << u, v;

                // 计算点到相机中心的距离
                float dist = pt_c.norm();

                // 如果需要保存深度图，更新深度图信息
                if(options_.if_save_depth_image){
                    // The distance of the lidar point from the center of the camera
                    auto iter = img.dist_map.find(uv);
                    if (iter != img.dist_map.end()){
                        // 如果已有深度值，取较小的深度值（前景优先）
                        iter->second = std::min(iter->second,dist);
                    } else {
                        // 如果没有深度值，添加新的深度值（需要加锁以保证线程安全）
                        proj_mutex_.lock();
                        img.dist_map.insert({uv,dist});
                        proj_mutex_.unlock();
                    }

                } 
                
                // 如果当前像素不是特征点，跳过后续处理
                if (img.feature_points.find(uv)==img.feature_points.end()) continue;
                // The distance of the lidar point from the center of the camera

                // 处理特征点的激光雷达对应关系
                auto iter = img.feature_pts_map.find(uv);
                if (iter!= img.feature_pts_map.end()){
                    // 如果特征点已有对应的激光点，仅当新点更近时更新
                    if (iter->second.second > dist){
                        iter->second = std::make_pair(pt,dist);
                    } else {continue;}
                } else {
                    // 如果特征点没有对应的激光点，添加新的对应关系
                    proj_mutex_.lock();
                    img.feature_pts_map.insert({uv,std::make_pair(pt,dist)});
                    proj_mutex_.unlock();
                }
                
            }
            }
        }
    }
}

void PcdProj::SaveDepthImage(const LImage& img){
    std::string folder_path = options_.original_image_folder + "/";
    std::string image_path = folder_path + img.img_name;
    cv::Mat original_image = cv::imread(image_path);

    cv::resize(original_image, original_image, cv::Size(img.img_width,img.img_height), 0, 0, cv::INTER_LINEAR);
    cv::Mat depth_image(img.img_height,img.img_width,CV_8UC3,cv::Scalar(255,255,255));
    float color_scale = 255/options_.choose_meter;
    for (auto iter : img.dist_map){
        
        int u = iter.first(0);
        int v = iter.first(1);
        depth_image.at<cv::Vec3b>(v,u)[0] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
        depth_image.at<cv::Vec3b>(v,u)[1] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
        depth_image.at<cv::Vec3b>(v,u)[2] = cv::saturate_cast<uint8_t>(static_cast<int>(iter.second * color_scale));
    }

    cv::Mat result_image(img.img_height,img.img_width,CV_8UC3);
    cv::addWeighted(depth_image,0.8,original_image,0.2,0,result_image);

    for(auto& point : img.feature_points){
		cv::circle(result_image, cv::Point(point(0),point(1)), 4, cv::Scalar(0, 255, 120), -1);
	}

    std::string result_image_path = options_.depth_image_folder + "/";
    std::string result_image_name = result_image_path + img.img_name;
    cv::imwrite(result_image_name,result_image);
}

void PcdProj::SearchImageMap(QuadPyramid& quad, ImageMapType& image_map){

    // 收集视锥体所有顶点和全局地图边界的坐标
    // 创建包含视锥体顶点和全局地图边界的x坐标数组
    std::vector<float> x{quad.vertex(0),quad.corner_1(0),quad.corner_2(0),quad.corner_3(0),
            quad.corner_4(0), global_map_min_x_,global_map_max_x_};
    std::vector<float> y{quad.vertex(1),quad.corner_1(1),quad.corner_2(1),quad.corner_3(1),
            quad.corner_4(1), global_map_min_y_,global_map_max_y_};
    std::vector<float> z{quad.vertex(2),quad.corner_1(2),quad.corner_2(2),quad.corner_3(2),
            quad.corner_4(2), global_map_min_z_,global_map_max_z_};

    // 对x、y、z坐标数组进行排序，以获取最小值和最大值
    std::sort(x.begin(),x.end());
    std::sort(y.begin(),y.end());
    std::sort(z.begin(),z.end());

    // 计算子地图索引的范围
    // 将空间中的实际坐标转换为子地图索引（网格索引）
    int x_min = round(x.front()/options_.submap_length);
    int x_max = round(x.back()/options_.submap_length);
    int y_min = round(y.front()/options_.submap_height);
    int y_max = round(y.back()/options_.submap_height);
    int z_min = round(z.front()/options_.submap_width);
    int z_max = round(z.back()/options_.submap_width);
    
    // 遍历视锥体可能覆盖的空间网格（子地图）
    // 每个方向都扩展1个网格，确保不遗漏相交的网格
    for(int idx = x_min-1; idx <= x_max+1; idx++){
        float x = idx * options_.submap_length; // 转换回实际坐标
        for(int idy = y_min-1; idy <= y_max+1; idy++){
            float y = idy * options_.submap_height; // 转换回实际坐标
            for(int idz = z_min-1; idz <= z_max+1; idz++){
                float z = idz * options_.submap_width; // 转换回实际坐标

                // 检查子地图中心点是否在视锥体内部
                // 视锥体由5个平面定义，判断点是否在所有平面的内侧
                // condition_0 到 condition_4 分别检查点是否在视锥体的5个面的内侧
                // plane_i 表示视锥体的第i个平面方程系数(ax+by+cz+d=0)
                // 点在平面内侧的条件是 ax+by+cz+d <= 0
                bool condition_0 = (quad.plane_0(0)*x + quad.plane_0(1)*y + quad.plane_0(2)*z + quad.plane_0(3) <= 0.0);
                bool condition_1 = (quad.plane_1(0)*x + quad.plane_1(1)*y + quad.plane_1(2)*z + quad.plane_1(3) <= 0.0);
                bool condition_2 = (quad.plane_2(0)*x + quad.plane_2(1)*y + quad.plane_2(2)*z + quad.plane_2(3) <= 0.0);
                bool condition_3 = (quad.plane_3(0)*x + quad.plane_3(1)*y + quad.plane_3(2)*z + quad.plane_3(3) <= 0.0);
                bool condition_4 = (quad.plane_4(0)*x + quad.plane_4(1)*y + quad.plane_4(2)*z + quad.plane_4(3) <= 0.0);

                // 如果点在视锥体内部（满足所有平面条件）
                if(condition_1 && condition_2 && condition_3 && condition_4 && condition_0){
                    // 构建子地图的键值
                    KeyType key;
                    key << idx, idy, idz; // 将网格索引作为键值

                    // 查找对应的子地图
                    auto iter = submap_.find(key);
                    // 如果找到子地图，将其地址添加到image_map中
                    if (iter != submap_.end()){image_map.push_back(&(iter->second));}
                }
            }
        } 
    }
    return ;
}

Eigen::Vector2d PcdProj::DistortOpenCV(Eigen::Vector2d& ori_uv, const Camera& camera){
    
    std::vector<double> params = camera.Params();
    double fx = params[0]; 
    double fy = params[1];
    double cx = params[2];
    double cy = params[3];
    double k1 = params[4];
    double k2 = params[5];
    double p1 = params[6];
    double p2 = params[7];

    // Normalization
    double x_corrected = (ori_uv(0) - cx) / fx;
    double y_corrected = (ori_uv(1) - cy) / fy;

    double r2 = x_corrected * x_corrected + y_corrected * y_corrected;
    double deltaRa = 1. + k1 * r2 + k2 * r2 * r2;
    double deltaRb = 1;
    double deltaTx = 2. * p1 * x_corrected * y_corrected + p2 * (r2 + 2. * x_corrected * x_corrected);
    double deltaTy = p1 * (r2 + 2. * y_corrected * y_corrected) + 2. * p2 * x_corrected * y_corrected;

    double distort_u0 = x_corrected * deltaRa * deltaRb + deltaTx;
    double distort_v0 = y_corrected * deltaRa * deltaRb + deltaTy;

    distort_u0 = distort_u0 * fx + cx;
    distort_v0 = distort_v0 * fy + cy;

    Eigen::Vector2d dis_uv;
    dis_uv << distort_u0, distort_v0;

    return dis_uv;

}

} //namespace lidar
} //namespace colmap
