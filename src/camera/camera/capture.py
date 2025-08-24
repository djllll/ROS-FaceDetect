import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    
    def __init__(self):
        super().__init__('camera_node')
        
        # 固定配置参数
        self.camera_device = 0  # 使用第一个摄像头
        self.image_topic = 'camera/image_raw'  # 发布的图像话题
        self.frame_id = 'camera_frame'  # 坐标系ID
        self.publish_rate = 10.0  # 发布频率（Hz）
        
        # 创建图像发布者
        self.image_publisher = self.create_publisher(Image, self.image_topic, 10)
        
        # 创建定时器，控制发布频率
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        
        # 创建OpenCV与ROS图像转换桥梁
        self.bridge = CvBridge()
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_device)
        
        # 检查摄像头是否成功打开
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开摄像头设备 {self.camera_device}")
            return
        
        self.get_logger().info(f"摄像头节点已启动，发布图像到话题: {self.image_topic}")
        
        # 图像计数
        self.image_count = 0

    def timer_callback(self):
        """定时器回调函数，捕获并发布图像"""
        if not self.cap.isOpened():
            return
            
        # 从摄像头捕获一帧图像
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().error("无法获取图像帧")
            return
        
        # 转换OpenCV图像到ROS图像消息并发布
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = self.frame_id
            self.image_publisher.publish(ros_image)
            
            # 每10帧打印一次日志
            self.image_count += 1
            if self.image_count % 10 == 0:
                self.get_logger().debug(f"已发布 {self.image_count} 帧图像")
                
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")

    def destroy_node(self):
        """节点销毁时释放摄像头资源"""
        if self.cap.isOpened():
            self.cap.release()
        self.get_logger().info("摄像头节点已关闭")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    