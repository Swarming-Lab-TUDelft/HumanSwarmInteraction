/* TU Delft Swarm Lab - Human Swarm Interaction Project
 * Description: Track and filter multiple keypoints linearly.
 * Author: Alexander James Becoy
 * Date: 2024-11-06
 */

#include <keypoints_tracking/multiple_keypoints_tracker_node.hpp>

using namespace HumanSwarmInteraction;

MultipleKeypointsTrackerNode::MultipleKeypointsTrackerNode()
: Node("multiple_keypoints_tracking_node")
{
  configureParameters();
  initializeKalmanFilters();
  initializeKeyPointsPublisher();
  initializeKeyPointsSubscriber();
  RCLCPP_INFO(get_logger(), "Multiple Keypoints Tracker Node initialized.");
}

MultipleKeypointsTrackerNode::~MultipleKeypointsTrackerNode()
{
  RCLCPP_WARN(get_logger(), "Multiple Keypoints Tracker Node destroyed.");
}

void MultipleKeypointsTrackerNode::timerCallback()
{
  // Get the centroids of the Kalman filters
  std::vector<std::array<double, OUTPUT_SIZE>> centroids;
  for (const auto& keypoint_name : m_keypoint_names) {
    // Skip if the Kalman filter has not been initialized yet
    if (!m_kalman_filters[keypoint_name]->isInitialized()) {
      continue;
    }
    centroids.push_back(m_kalman_filters[keypoint_name]->getCentroidCoordinate());
  }
  publishKeypoints(centroids);
}

void MultipleKeypointsTrackerNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Received image message.");
  std::vector<std::array<double, OUTPUT_SIZE>> centroids;
  for (const auto& keypoint_name : m_keypoint_names) {
    if (!m_kalman_filters[keypoint_name]->isInitialized()) {
      continue;
    }
    centroids.push_back(m_kalman_filters[keypoint_name]->getCentroidCoordinate());
  }
  publishKeypoints(centroids);
}

void MultipleKeypointsTrackerNode::keypointsCallback(
  const human_swarm_interaction_interfaces::msg::PoseKeypointsStamped::SharedPtr msg)
{
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Received keypoints message.");
  try {
    for (auto& keypoint : msg->keypoints) {
      // Skip if the keypoint is not in the list of keypoint names
      if (std::find(m_keypoint_names.begin(), m_keypoint_names.end(), keypoint.name) == m_keypoint_names.end()) {
        continue;
      }

      // Skip if the keypoint confidence is below a threshold
      if (keypoint.confidence < 0.5) {
        continue;
      }

      double width = m_tracking_window_width * keypoint.confidence;
      double height = m_tracking_window_height * keypoint.confidence;

      if (m_kalman_filters[keypoint.name]->isInitialized()) {
        m_kalman_filters[keypoint.name]->predict();
        m_kalman_filters[keypoint.name]->update(keypoint.x, keypoint.y, width, height);
      } else {
        m_kalman_filters[keypoint.name]->setInitialState(
          keypoint.x, keypoint.y, width, height, 0.0, 0.0, 0.0, 0.0);
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
      "Error in keypoint tracking: %s", e.what());
  }
}

void MultipleKeypointsTrackerNode::publishKeypoints(
  const std::vector<std::array<double,OUTPUT_SIZE>>& centroids) const
{
  human_swarm_interaction_interfaces::msg::PoseKeypointsStamped msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "camera_link";  // TODO: Should get this from the input message

  for (size_t i = 0; i < centroids.size(); i++) {
    human_swarm_interaction_interfaces::msg::PoseKeypoint keypoint;
    keypoint.name = m_keypoint_names[i];
    keypoint.x = centroids[i][0];
    keypoint.y = centroids[i][1];
    msg.keypoints.push_back(keypoint);
  }

  m_keypoints_pub->publish(msg);
}

void MultipleKeypointsTrackerNode::initializeKalmanFilters()
{
  // Get the parameters for the Kalman Filter
  this->declare_parameter("initial_state", std::vector<double>());
  this->declare_parameter("covariances", std::vector<double>());
  this->declare_parameter("process_noises", std::vector<double>());
  this->declare_parameter("observation_noises", std::vector<double>());

  std::vector<double> initial_state = this->get_parameter("initial_state").as_double_array();
  std::vector<double> covariances = this->get_parameter("covariances").as_double_array();
  std::vector<double> process_noises = this->get_parameter("process_noises").as_double_array();
  std::vector<double> observation_noises = this->get_parameter("observation_noises").as_double_array();

  // Check if the parameters are valid
  if (initial_state.size() != STATE_SIZE) {
    throw std::invalid_argument("Initial state must have size 8. Current size: " 
      + std::to_string(initial_state.size()));
  }

  if (covariances.size() != STATE_SIZE) {
    throw std::invalid_argument("Covariances must have size 8. Current size: " 
      + std::to_string(covariances.size()));
  }

  if (process_noises.size() != STATE_SIZE) {
    throw std::invalid_argument("Process noises must have size 8. Current size: " 
      + std::to_string(process_noises.size()));
  }

  if (observation_noises.size() != MEASUREMENT_SIZE) {
    throw std::invalid_argument("Observation noises must have size 4. Current size: " 
      + std::to_string(observation_noises.size()));
  }

  // Initialize timer
  this->declare_parameter("timer_period", 0.01);
  double timer_period = this->get_parameter("timer_period").as_double();

  m_timer = this->create_wall_timer(std::chrono::milliseconds((int)(timer_period * 1000)), 
    std::bind(&MultipleKeypointsTrackerNode::timerCallback, this));
  RCLCPP_INFO(get_logger(), "Timer initialized with period: %f", timer_period);

  // Initialize Kalman filters
  for (const auto& keypoint_name : m_keypoint_names) {
    m_kalman_filters[keypoint_name] = std::make_shared<LinearKalmanFilter>(
      initial_state, covariances, process_noises, observation_noises, timer_period);
  }

  RCLCPP_INFO(get_logger(), "Kalman filters initialized for %d keypoints", m_kalman_filters.size());
}

void MultipleKeypointsTrackerNode::initializeImageSubscriber()
{
  this->declare_parameter("input_image_topic_name", "image_raw");
  this->declare_parameter("input_image_topic_queue_size", 10);

  std::string input_image_topic_name = this->get_parameter("input_image_topic_name").as_string();
  int input_image_topic_queue_size = this->get_parameter("input_image_topic_queue_size").as_int();
  RCLCPP_INFO(get_logger(), "Subscribing to topic: %s with queue size: %d", 
    input_image_topic_name.c_str(), input_image_topic_queue_size);

  m_image_sub = this->create_subscription<sensor_msgs::msg::Image>(
    input_image_topic_name, input_image_topic_queue_size, 
    std::bind(&MultipleKeypointsTrackerNode::imageCallback, this, std::placeholders::_1));
}

void MultipleKeypointsTrackerNode::initializeKeyPointsSubscriber()
{
  this->declare_parameter("input_keypoints_topic_name", "all_keypoints");
  this->declare_parameter("input_keypoints_topic_queue_size", 10);

  std::string input_keypoints_topic_name = this->get_parameter("input_keypoints_topic_name").as_string();
  int input_keypoints_topic_queue_size = this->get_parameter("input_keypoints_topic_queue_size").as_int();
  RCLCPP_INFO(get_logger(), "Subscribing to topic: %s with queue size: %d", 
    input_keypoints_topic_name.c_str(), input_keypoints_topic_queue_size);

  m_keypoints_sub = this->create_subscription<human_swarm_interaction_interfaces::msg::PoseKeypointsStamped>(
    input_keypoints_topic_name, input_keypoints_topic_queue_size, 
    std::bind(&MultipleKeypointsTrackerNode::keypointsCallback, this, std::placeholders::_1));
}

void MultipleKeypointsTrackerNode::initializeKeyPointsPublisher()
{
  this->declare_parameter("output_keypoints_topic_name", "filtered_keypoints");
  this->declare_parameter("output_keypoints_topic_queue_size", 10);

  std::string output_keypoints_topic_name = this->get_parameter("output_keypoints_topic_name").as_string();
  int output_keypoints_topic_queue_size = this->get_parameter("output_keypoints_topic_queue_size").as_int();
  RCLCPP_INFO(get_logger(), "Publishing to topic: %s with queue size: %d", 
    output_keypoints_topic_name.c_str(), output_keypoints_topic_queue_size);

  m_keypoints_pub = this->create_publisher<human_swarm_interaction_interfaces::msg::PoseKeypointsStamped>(
    output_keypoints_topic_name, output_keypoints_topic_queue_size);
}

void MultipleKeypointsTrackerNode::configureParameters()
{
  // Get the keypoint names to track
  this->declare_parameter("keypoint_names", std::vector<std::string>{"right_wrist", "left_wrist"});
  m_keypoint_names = this->get_parameter("keypoint_names").as_string_array();

  // Concatenate the keypoint names into a single string for logging
  std::string keypoint_names_str = "";
  for (const auto& keypoint_name : m_keypoint_names) {
    keypoint_names_str += keypoint_name;
    
    // Append a comma if the keypoint name is not the last one
    if (keypoint_name != m_keypoint_names.back()) {
      keypoint_names_str += ", ";
    }
  }
  RCLCPP_INFO(get_logger(), "Selected keypoint names: %s", keypoint_names_str.c_str());
  
  // Parameterize the tracking window dimensions
  this->declare_parameter("tracking_window_width", 0.1);
  this->declare_parameter("tracking_window_height", 0.1);
  m_tracking_window_width = this->get_parameter("tracking_window_width").as_double();
  m_tracking_window_height = this->get_parameter("tracking_window_height").as_double();
  RCLCPP_INFO(get_logger(), "Tracking window width: %f, height: %f", 
    m_tracking_window_width, m_tracking_window_height);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MultipleKeypointsTrackerNode>());
  rclcpp::shutdown();
  return 0;
}