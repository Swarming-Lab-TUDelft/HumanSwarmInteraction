/* TU Delft Swarm Lab - Human Swarm Interaction Project
 * Description: Track and filter multiple keypoints linearly.
 * Author: Alexander James Becoy
 * Date: 2024-11-06
 */

#ifndef KEYPOINTS_TRACKING__MULTIPLE_KEYPOINTS_TRACKER_NODE_HPP_
#define KEYPOINTS_TRACKING__MULTIPLE_KEYPOINTS_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <human_swarm_interaction_interfaces/msg/pose_keypoints_stamped.hpp>

#include <vector>
#include <string>
#include <unordered_map>

#include <keypoints_tracking/linear_kalman_filter.hpp>

namespace HumanSwarmInteraction
{
  class MultipleKeypointsTrackerNode : public rclcpp::Node
  {
    public:
      MultipleKeypointsTrackerNode();
      ~MultipleKeypointsTrackerNode();

    private:
      void timerCallback();
      void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
      void keypointsCallback(const human_swarm_interaction_interfaces::msg::PoseKeypointsStamped::SharedPtr msg);
      void publishKeypoints(const std::vector<std::array<double,OUTPUT_SIZE>>& centroids) const;

      void initializeKalmanFilters();
      void initializeImageSubscriber();
      void initializeKeyPointsSubscriber();
      void initializeKeyPointsPublisher();
      void configureParameters();

      std::unordered_map<std::string, LinearKalmanFilter::SharedPtr> m_kalman_filters;
      std::vector<std::string> m_keypoint_names;
      double m_tracking_window_width;
      double m_tracking_window_height;

      rclcpp::TimerBase::SharedPtr m_timer;
      rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_image_sub;
      rclcpp::Subscription<human_swarm_interaction_interfaces::msg::PoseKeypointsStamped>::SharedPtr m_keypoints_sub;
      rclcpp::Publisher<human_swarm_interaction_interfaces::msg::PoseKeypointsStamped>::SharedPtr m_keypoints_pub;

  };  // class MultipleKeypointsTrackerNode
} // namespace HumanSwarmInteraction

#endif  // KEYPOINTS_TRACKING__MULTIPLE_KEYPOINTS_TRACKER_NODE_HPP_
