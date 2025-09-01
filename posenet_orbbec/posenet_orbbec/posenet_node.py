import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from human_swarm_interaction_interfaces.msg import PoseKeypointsStamped, PoseKeypoint

SKELETON_CONNECTIONS = [
    (5, 6),  # Shoulders
    (5, 7),  # Left arm (shoulder to elbow)
    (7, 9),  # Left arm (elbow to wrist)
    (6, 8),  # Right arm (shoulder to elbow)
    (8, 10), # Right arm (elbow to wrist)
    (5, 11), # Left hip to left shoulder
    (6, 12), # Right hip to right shoulder
    (11, 12),# Hips
    (11, 13),# Left leg (hip to knee)
    (13, 15),# Left leg (knee to ankle)
    (12, 14),# Right leg (hip to knee)
    (14, 16) # Right leg (knee to ankle)
]
KEYPOINT_LABELS = [
    "nose",         # 0
    "left_eye",     # 1
    "right_eye",    # 2
    "left_ear",     # 3
    "right_ear",    # 4
    "left_shoulder",# 5
    "right_shoulder",# 6
    "left_elbow",   # 7
    "right_elbow",  # 8
    "left_wrist",   # 9
    "right_wrist",  # 10
    "left_hip",     # 11
    "right_hip",    # 12
    "left_knee",    # 13
    "right_knee",   # 14
    "left_ankle",   # 15
    "right_ankle"   # 16
]

class PoseNetNode(Node):
    def __init__(self):
        super().__init__('posenet_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listner_callback,
            10)
        self.publisher = self.create_publisher(Image,  '/posenet/image_annotated', 10)
        self.keypoints_publisher = self.create_publisher(PoseKeypointsStamped, '/posenet/keypoints', 10)
        self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = self.model.signatures['serving_default']
        self.input_size = 192
        self.bridge = CvBridge()

    def listner_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        input_image = self.preprocess_image(cv_image)
        all_keypoints = self.run_pose_estimation(input_image)
        filtered_keypoints = self.filter_keypoints(all_keypoints)
        self.publish_keypoints(filtered_keypoints)

        annotated_img = self.draw_keypoints(cv_image, all_keypoints)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='rgb8')
        self.publisher.publish(annotated_msg)


    def preprocess_image(self, image):
        """Preprocess the image for pose estimation: resizing and normalizing."""
        image_resized = tf.image.resize_with_pad(image, self.input_size, self.input_size)
        input_image = tf.cast(image_resized, dtype=tf.int32)
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        return input_image        

    def run_pose_estimation(self, input_image):
        input_tensor = tf.convert_to_tensor(input_image, dtype=tf.int32)
        input_tensor = tf.reshape(input_tensor, (1, 192, 192, 3))
        outputs = self.movenet(input_tensor)
        keypoints = outputs['output_0'].numpy()[0]  # Extract keypoints
        return keypoints
    
    def filter_keypoints(self, keypoints, threshold=0.5):
        valid_keypoints = []
        for idx, (_, _, confidence) in enumerate(keypoints[0]):
            if confidence < threshold:
                continue
            valid_keypoints.append(keypoints[0][idx])
        return valid_keypoints
    
    def draw_keypoints(self, image, keypoints, threshold=0.5):
        """Draw keypoints and skeleton on the image."""
        height, width, _ = image.shape
        
        # List to store valid points (with confidence > threshold)
        valid_points = []

        # Iterate over the keypoints to draw circles at detected joints
        for idx, kp in enumerate(keypoints[0]):
            y, x, confidence = kp
            if confidence > threshold:
                cx, cy = int(x * width), int(y * height)
                valid_points.append((cx, cy))
                cv2.circle(image, (cx, cy), 4, (0, 255, 0), -3)

                # Put the label next to the keypoint
                label = KEYPOINT_LABELS[idx]
                cv2.putText(image, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)


        # Draw skeleton lines based on the connections
        for connection in SKELETON_CONNECTIONS:
            point1 = connection[0]
            point2 = connection[1]
            if point1 < len(valid_points) and point2 < len(valid_points):
                if keypoints[0][point1][2] > threshold and keypoints[0][point2][2] > threshold:
                    # Draw line between valid points
                    cv2.line(image, valid_points[point1], valid_points[point2], (0, 255, 255), 6)

        return image

    def publish_keypoints(self, keypoints):
        # Do not publish if the list of keypoints is empty
        if len(keypoints) == 0:
            return

        self.get_logger().info('Publishing %d keyponts...' % len(keypoints), throttle_duration_sec=5)
        keypoints_msg = PoseKeypointsStamped()
        keypoints_msg.header.frame_id = 'camera_frame'
        keypoints_msg.header.stamp = self.get_clock().now().to_msg()

        for idx, (y, x, confidence) in enumerate(keypoints):
            keypoint = PoseKeypoint()
            keypoint.name = KEYPOINT_LABELS[idx]
            keypoint.x = float(x)
            keypoint.y = float(y)
            keypoint.confidence = float(confidence)
            keypoints_msg.keypoints.append(keypoint)

        self.keypoints_publisher.publish(keypoints_msg)

def main(args=None):
    rclpy.init(args=args)
    posenet_node = PoseNetNode()
    rclpy.spin(posenet_node)
    posenet_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()