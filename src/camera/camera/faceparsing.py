import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort
import numpy as np
from PIL import Image as PILImage
import cv2

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        
        # 模型配置 - 确保这些路径是正确的字符串
        self.onnx_path = "model_quantized.onnx"  # 模型路径
        self.input_size = (512, 512)  # 确保这是整数元组
        
        # 加载模型
        self.session = ort.InferenceSession(self.onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 确定类别数量（确保是整数）
        self.num_classes = 14
        self.get_logger().info(f"类别数量: {self.num_classes}")
        
        # 创建颜色映射
        self.color_map = self.create_color_map()
        
        # ROS设置
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Image, 'segmentation_result', 10)

    def create_color_map(self):
        """创建与类别数量匹配的颜色映射"""
        color_map = []
        for i in range(self.num_classes):
            # 使用简单的算法生成不同颜色
            color_map.append([
                (i * 100) % 256,
                (i * 150) % 256,
                (i * 200) % 256
            ])
        return np.array(color_map, dtype=np.uint8)

    def callback(self, msg):
        try:
            # 1. 将ROS图像转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 获取原始尺寸（确保是整数元组）
            orig_height, orig_width = cv_image.shape[:2]
            orig_size = (orig_width, orig_height)
            
            # 2. 预处理
            # 转换为RGB并调整大小
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            img_resized = pil_img.resize(self.input_size)  # 使用整数尺寸
            
            # 转换为模型输入格式
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            img_np = np.transpose(img_np, (2, 0, 1))
            img_np = np.expand_dims(img_np, axis=0)
            
            # 3. 推理
            outputs = self.session.run([self.output_name], {self.input_name: img_np})
            logits = outputs[0]
            
            # 4. 处理结果
            labels = np.argmax(logits, axis=1)[0].astype(np.uint8)  # 确保是整数类型
            
            # 调整回原始尺寸
            labels_resized = PILImage.fromarray(labels).resize(orig_size, PILImage.NEAREST)
            labels_viz = np.array(labels_resized, dtype=np.int32)  # 明确为整数类型
            
            # 应用颜色映射
            labels_viz = np.clip(labels_viz, 0, self.num_classes - 1)  # 确保索引有效
            colored_result = self.color_map[labels_viz]
            
            # 5. 发布结果
            result_msg = self.bridge.cv2_to_imgmsg(colored_result, 'rgb8')
            result_msg.header = msg.header
            self.pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f'处理错误: {str(e)}')
            
def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    
    # 使用多线程执行器
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
