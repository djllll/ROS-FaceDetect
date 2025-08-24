import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from PIL import Image as PILImage

class SegmentationAlgorithm:
    def __init__(self, onnx_path, input_size=(512, 512), num_classes=14):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = input_size
        self.num_classes = num_classes
        self.color_map = self.create_color_map()

    def create_color_map(self):
        color_map = []
        for i in range(self.num_classes):
            color_map.append([
                (i * 100) % 256,
                (i * 150) % 256,
                (i * 200) % 256
            ])
        return np.array(color_map, dtype=np.uint8)

    def process(self, cv_image):
        orig_height, orig_width = cv_image.shape[:2]
        orig_size = (orig_width, orig_height)
        
        pil_img = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        img_resized = pil_img.resize(self.input_size)
        
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)
        
        outputs = self.session.run([self.output_name], {self.input_name: img_np})
        logits = outputs[0]
        
        labels = np.argmax(logits, axis=1)[0].astype(np.uint8)
        
        labels_resized = PILImage.fromarray(labels).resize(orig_size, PILImage.NEAREST)
        labels_viz = np.array(labels_resized, dtype=np.int32)
        
        labels_viz = np.clip(labels_viz, 0, self.num_classes - 1)
        colored_result = self.color_map[labels_viz]
        
        return colored_result

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        
        self.onnx_path = "model_quantized.onnx"
        self.input_size = (512, 512)
        self.num_classes = 14
        
        self.algorithm = SegmentationAlgorithm(
            self.onnx_path,
            self.input_size,
            self.num_classes
        )
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, 
            'camera/image_raw', 
            self.callback, 
            10
        )
        self.pub = self.create_publisher(Image, 'segmentation_result', 10)
        self.get_logger().info("分割节点已启动")

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            result = self.algorithm.process(cv_image)
            
            result_msg = self.bridge.cv2_to_imgmsg(result, 'rgb8')
            result_msg.header = msg.header
            self.pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f'处理错误: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    
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
