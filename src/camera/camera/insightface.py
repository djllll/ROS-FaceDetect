import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster
import math
from insightface.app import FaceAnalysis

# 自定义下颌线索引（全局常量，算法与节点共用）
JAW_LINE_IDX = (
    [1]
    + list(range(9, 17))
    + list(range(2, 9))
    + [0]
    + list(range(24, 17, -1))
    + list(range(32, 24, -1))
    + [17]
)


class FaceDetectionAlgorithm:
    """
    独立的人脸检测算法类（无ROS依赖，可移植到其他平台）
    包含：模型初始化、人脸检测、特征点提取、掩码生成、绘图等核心逻辑
    """
    def __init__(self, det_size=(640, 640)):
        self.det_size = det_size  # 模型输入尺寸
        self.app = None  # InsightFace模型实例
        self.initialize_model()  # 初始化模型

    def initialize_model(self):
        """初始化InsightFace模型（支持2D 106点特征点）"""
        try:
            self.app = FaceAnalysis(
                name="antelopev2", 
                providers=["CPUExecutionProvider"]  # CPU推理（可换GPU）
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
        except Exception as e:
            raise Exception(f"模型初始化失败: {str(e)}")  # 抛出异常由调用方处理

    def detect_faces(self, frame):
        """检测图像中的人脸，返回人脸信息列表"""
        return self.app.get(frame) if self.app is not None else []

    def get_2d_106_landmarks(self, face):
        """从人脸信息中提取2D 106点特征点"""
        return face["landmark_2d_106"] if "landmark_2d_106" in face else None

    def get_face_contour_points(self, landmarks_2d):
        """基于106点特征点提取人脸轮廓关键点（使用自定义下颌线索引）"""
        return [landmarks_2d[i] for i in JAW_LINE_IDX]

    def draw_2d_face_contour(self, frame, contour_points, color=(0, 255, 0), thickness=2):
        """在图像上绘制人脸轮廓"""
        if len(contour_points) < 3:
            return frame  # 轮廓点不足时直接返回原帧
        
        contour_points = np.array(contour_points, dtype=np.int32)
        points_num = len(contour_points)
        
        # 绘制轮廓线段
        for i in range(points_num - 1):
            x1, y1 = contour_points[i]
            x2, y2 = contour_points[i + 1]
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 闭合轮廓（最后一点连回起点）
        x1, y1 = contour_points[-1]
        x2, y2 = contour_points[0]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    def extract_skin_mask(self, face_img):
        """提取皮肤区域掩码（基于YCbCr颜色空间，滤除黑色区域）"""
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        
        # 滤除黑色区域（Y通道值过低视为黑色）
        non_black_mask = y_channel > 30
        cr_mask = (cr_channel >= 133) & (cr_channel <= 173)
        cb_mask = (cb_channel >= 77) & (cb_channel <= 127)
        
        # 合并掩码并转换为二值图（0=非皮肤，255=皮肤）
        mask = non_black_mask & cr_mask & cb_mask
        mask = mask.astype(np.uint8) * 255
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除噪点
        return mask

    def get_jawline_mask(self, frame, landmarks_2d, bbox):
        """基于下颌线生成人脸内部掩码（排除下颌线以下区域）"""
        x1, y1, x2, y2 = bbox  # 人脸 bounding box
        jawline_points = landmarks_2d[JAW_LINE_IDX].astype(np.int32)  # 下颌线关键点
        
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = 255  # bbox内为候选区域
        
        jaw_below = np.vstack([
            jawline_points,
            np.array([[x2, y2]], dtype=np.int32),  # bbox右下角
            np.array([[x1, y2]], dtype=np.int32)   # bbox左下角
        ])
        
        cv2.fillPoly(full_mask, [jaw_below], 0)
        return full_mask

    def get_binary_skin_mask(self, frame, bbox, landmarks_2d):
        """获取二值化皮肤掩码（bbox内 + 下颌线以上 + 肤色区域）"""
        x1, y1, x2, y2 = bbox
        face_roi = frame[y1:y2, x1:x2]  # 裁剪人脸ROI
        
        if face_roi.size == 0:
            return None  # ROI为空时返回None
        
        # 1. 提取ROI内的皮肤掩码
        skin_mask_roi = self.extract_skin_mask(face_roi)
        # 2. 还原掩码到原图尺寸
        skin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        skin_mask[y1:y2, x1:x2] = skin_mask_roi
        # 3. 结合下颌线掩码（仅保留人脸内部皮肤）
        jaw_mask = self.get_jawline_mask(frame, landmarks_2d, bbox)
        final_mask = cv2.bitwise_and(skin_mask, jaw_mask)
        
        return final_mask

    @staticmethod
    def euler_to_quaternion(pitch, yaw, roll):
        """欧拉角（俯仰/偏航/滚转）转换为四元数（ROS标准右手坐标系）"""
        # 角度转弧度
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)
        
        # 四元数计算（参考ROS TF标准）
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        # 封装为ROS Quaternion消息
        quat = Quaternion()
        quat.w = w
        quat.x = x
        quat.y = y
        quat.z = z
        return quat


class FaceDetectionNode(Node):
    """
    ROS2人脸检测节点（仅负责ROS通信，不包含核心算法）
    包含：话题订阅/发布、TF广播、帧缓冲与线程管理
    """
    def __init__(self):
        super().__init__("face_detection_node")
        
        # 1. 初始化算法实例（无ROS依赖，可独立配置）
        try:
            self.algorithm = FaceDetectionAlgorithm(det_size=(640, 640))
        except Exception as e:
            self.get_logger().fatal(f"算法初始化失败: {str(e)}")
            raise  # 终止节点启动
        
        self.processing = False  # 标记是否正在处理帧
        self.latest_frame = None  # 缓存最新帧（避免丢帧）
        self.lock = threading.Lock()  # 线程锁（保护共享变量）
        
        self.bridge = CvBridge()  # ROS Image与OpenCV格式转换
        self.tf_broadcaster = TransformBroadcaster(self)  # TF变换广播
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # 尽力而为（适合实时流）
            history=HistoryPolicy.KEEP_LAST,            # 仅保留最新帧
            depth=1                                     # 缓冲区深度1
        )
        
        self.image_sub = self.create_subscription(
            Image, 
            "camera/image_raw",  # 输入话题（需与摄像头节点匹配）
            self.image_callback, 
            qos_profile
        )
        self.result_pub = self.create_publisher(
            Image, 
            "face_detection/result",  # 输出结果话题
            10  # 队列大小
        )
        self.skin_mask_pub = self.create_publisher(
            Image, 
            "face_detection/skin_mask",  # 皮肤掩码话题
            10
        )
        
        threading.Thread(target=self.process_frame, daemon=True).start()
        
        self.get_logger().info("人脸检测节点已启动（算法与ROS通信分离）")

    def image_callback(self, msg):
        """ROS图像话题回调：缓存最新帧（避免处理延迟导致的帧堆积）"""
        with self.lock:  # 线程锁保护共享变量
            try:
                # 将ROS Image消息转换为OpenCV格式（BGR8）
                self.latest_frame = self.bridge.imgmsg_to_cv2(
                    msg, 
                    desired_encoding="bgr8"
                )
            except Exception as e:
                self.get_logger().error(f"图像格式转换失败: {str(e)}")

    def process_frame(self):
        while rclpy.ok(): 
            with self.lock:
                if self.latest_frame is None or self.processing:
                    continue  
                frame = self.latest_frame.copy()
                self.latest_frame = None  # 清空缓存，等待新帧
                self.processing = True     # 标记为处理中
            
            # 检查算法模型是否就绪
            if self.algorithm.app is None:
                self.get_logger().warn("模型未就绪，跳过帧处理")
                with self.lock:
                    self.processing = False
                continue
            
            try:
                faces = self.algorithm.detect_faces(frame)
                binary_skin_window = np.zeros_like(frame[:, :, 0])
                
                if faces:
                    face = faces[0]
                    bbox = face["bbox"].astype(int)
                    cv2.rectangle(
                        frame, 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        (255, 0, 0),  # 蓝色框
                        1             # 线宽
                    )
                    
                    # 3. 提取2D 106点特征点并处理
                    landmarks_2d = self.algorithm.get_2d_106_landmarks(face)
                    if landmarks_2d is None:
                        # 无106点特征点时绘制提示
                        cv2.putText(
                            frame,
                            "不支持2D 106点特征点",
                            (bbox[0], bbox[1] - 10),  # 文字位置（bbox上方）
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,        # 字体大小
                            (0, 0, 255),# 红色文字
                            1           # 文字线宽
                        )
                    else:
                        # 生成皮肤掩码
                        skin_mask = self.algorithm.get_binary_skin_mask(
                            frame, bbox, landmarks_2d
                        )
                        if skin_mask is not None:
                            binary_skin_window = skin_mask
                        
                        # 绘制人脸轮廓
                        contour_points = self.algorithm.get_face_contour_points(landmarks_2d)
                        frame = self.algorithm.draw_2d_face_contour(frame, contour_points)
                        
                        # 4. 计算人脸姿态并广播TF变换
                        pitch, yaw, roll = face["pose"]  # 欧拉角姿态
                        transform_msg = TransformStamped()
                        # 设置TF头部信息
                        transform_msg.header.stamp = self.get_clock().now().to_msg()
                        transform_msg.header.frame_id = "world"  # 父坐标系
                        transform_msg.child_frame_id = "face"   # 子坐标系（人脸）
                        # 设置平移（基于bbox中心，除以200缩放为合理尺寸）
                        x1, y1, x2, y2 = bbox
                        transform_msg.transform.translation.x = (x1 + x2) / 2 / 200
                        transform_msg.transform.translation.y = (y1 + y2) / 2 / 200
                        transform_msg.transform.translation.z = 0.0  # 简化为2D，z轴设为0
                        # 欧拉角转四元数（调用算法类静态方法）
                        transform_msg.transform.rotation = self.algorithm.euler_to_quaternion(
                            pitch, yaw, roll
                        )
                        # 广播TF变换
                        self.tf_broadcaster.sendTransform(transform_msg)
                
                # 发布带检测标记的图像
                result_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.result_pub.publish(result_msg)
                # 发布皮肤掩码图像（单通道mono8格式）
                skin_mask_msg = self.bridge.cv2_to_imgmsg(binary_skin_window, encoding="mono8")
                self.skin_mask_pub.publish(skin_mask_msg)
                
            except Exception as e:
                self.get_logger().error(f"帧处理异常: {str(e)}")
            
            # 标记处理完成（释放锁）
            with self.lock:
                self.processing = False


def main(args=None):
    """ROS2节点入口函数"""
    rclpy.init(args=args)  
    node = FaceDetectionNode()  
    
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        node.get_logger().info("收到退出信号，正在关闭节点...")
    finally:
        node.destroy_node() 
        if rclpy.ok():
            rclpy.shutdown()  


if __name__ == "__main__":
    main()