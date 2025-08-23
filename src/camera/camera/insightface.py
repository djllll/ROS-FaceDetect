import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import threading
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Quaternion
import math

from tf2_ros.transform_broadcaster import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# 自定义下颌线索引
JAW_LINE_IDX = (
    [1]
    + list(range(9, 17))
    + list(range(2, 9))
    + [0]
    + list(range(24, 17, -1))
    + list(range(32, 24, -1))
    + [17]
)


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__("face_detection_node")

        # 性能优化参数
        self.processing = False  # 防止帧处理叠加
        self.latest_frame = None  # 存储最新帧
        self.lock = threading.Lock()  # 线程锁

        # 创建CV桥接器
        self.bridge = CvBridge()

        # 初始化模型 (在单独线程中初始化，避免阻塞节点)
        self.app = None
        threading.Thread(target=self.initialize_model, daemon=True).start()

        # 设置QoS配置，根据网络情况调整
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 创建订阅者和发布者
        self.image_sub = self.create_subscription(
            Image,
            "camera/image_raw",  # 订阅摄像头图像话题
            self.image_callback,
            qos_profile,
        )

        self.result_pub = self.create_publisher(
            Image, "face_detection/result", 10  # 发布处理结果话题
        )

        self.skin_mask_pub = self.create_publisher(
            Image, "face_detection/skin_mask", 10  # 发布皮肤掩码话题
        )

        # TF发布器
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("人脸检测节点已启动")

    def initialize_model(self):
        """初始化支持2D 106点特征点的模型"""
        try:
            self.app = FaceAnalysis(
                name="antelopev2", providers=["CPUExecutionProvider"]
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.get_logger().info("模型初始化完成")
        except Exception as e:
            self.get_logger().error(f"模型初始化失败: {str(e)}")

    def get_2d_106_landmarks(self, face):
        """提取2D 106点特征点"""
        if "landmark_2d_106" in face:
            return face["landmark_2d_106"]  # 直接获取106点特征点
        else:
            return None

    def get_face_contour_points(self, landmarks_2d):
        """基于2D 106点提取轮廓关键点"""
        contour_indices = JAW_LINE_IDX
        return [landmarks_2d[i] for i in contour_indices]

    def draw_2d_face_contour(
        self, frame, contour_points, color=(0, 255, 0), thickness=2
    ):
        """绘制基于2D 106点的人脸轮廓"""
        if len(contour_points) < 3:
            return frame

        contour_points = np.array(contour_points, dtype=np.int32)
        points_num = len(contour_points)

        # 绘制轮廓线条
        for i in range(points_num - 1):
            x1, y1 = contour_points[i]
            x2, y2 = contour_points[i + 1]
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        # 闭合轮廓
        x1, y1 = contour_points[-1]
        x2, y2 = contour_points[0]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        return frame

    def extract_skin_mask(self, face_img):
        """提取皮肤区域掩码（通过滤除黑色部分实现）"""
        # 转换到YCbCr颜色空间
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

        # 分离通道
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

        # 1. 滤除黑色区域：Y通道值过低的区域视为黑色
        non_black_mask = y_channel > 30

        # 2. 结合肤色的色度范围约束
        cr_mask = (cr_channel >= 133) & (cr_channel <= 173)
        cb_mask = (cb_channel >= 77) & (cb_channel <= 127)

        # 合并掩码：非黑色 + 符合肤色色度范围
        mask = non_black_mask & cr_mask & cb_mask

        # 转换为uint8类型掩码（0和255）
        mask = mask.astype(np.uint8) * 255

        # 形态学优化：去除噪点和空洞
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充小空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去除小噪点

        return mask

    def get_jawline_mask(self, frame, landmarks_2d, bbox):
        """基于106点的下颌线生成人脸内部掩码"""
        x1, y1, x2, y2 = bbox
        jawline_points = landmarks_2d[JAW_LINE_IDX].astype(np.int32)

        # 创建覆盖bbox的全白掩码
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = 255  # bbox内为候选区域

        # 生成下颌线以下的排除区域（下颌线+bbox底部两点）
        jaw_below = np.vstack(
            [
                jawline_points,
                np.array([[x2, y2]], dtype=np.int32),  # bbox右下角
                np.array([[x1, y2]], dtype=np.int32),  # bbox左下角
            ]
        )

        # 排除下颌线以下区域，保留下方为人脸内部
        cv2.fillPoly(full_mask, [jaw_below], 0)
        return full_mask

    def get_bbox_skin_mask(self, frame, bbox, landmarks_2d):
        """提取bbox内、下颌线以上的皮肤区域（基于106点）"""
        x1, y1, x2, y2 = bbox
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        # 提取皮肤掩码（已滤除黑色部分）
        skin_mask_roi = self.extract_skin_mask(face_roi)
        skin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        skin_mask[y1:y2, x1:x2] = skin_mask_roi

        # 结合下颌线掩码（基于106点）
        jaw_mask = self.get_jawline_mask(frame, landmarks_2d, bbox)
        final_mask = cv2.bitwise_and(skin_mask, jaw_mask)
        return final_mask

    def get_binary_skin_mask(self, frame, bbox, landmarks_2d):
        """获取二值化皮肤掩码（供单独窗口显示）"""
        return self.get_bbox_skin_mask(frame, bbox, landmarks_2d)

    def euler_to_quaternion(self, pitch, yaw, roll):
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)

        # 第二步：计算四元数（参考ROS标准右手坐标系）
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

        # 第三步：返回ROS Quaternion消息
        quat = Quaternion()
        quat.w = w
        quat.x = x
        quat.y = y
        quat.z = z
        return quat

    def process_frame(self):
        """处理图像帧的函数，在单独线程中运行"""
        while rclpy.ok():
            # 检查是否有新帧且不在处理中
            with self.lock:
                if self.latest_frame is None or self.processing:
                    continue

                # 获取最新帧并标记为处理中
                frame = self.latest_frame.copy()
                self.latest_frame = None
                self.processing = True

            # 确保模型已初始化
            if self.app is None:
                with self.lock:
                    self.processing = False
                continue

            try:
                # 镜像处理
                frame = cv2.flip(frame, 1)

                # 检测人脸 (性能优化：只处理必要的内容)
                faces = self.app.get(frame)
                binary_skin_window = np.zeros_like(frame[:, :, 0])  # 二值图窗口初始化

                face = faces[0]

                bbox = face["bbox"].astype(int)
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1
                )

                # 获取2D 106点特征点
                landmarks_2d = self.get_2d_106_landmarks(face)
                if landmarks_2d is None:
                    cv2.putText(
                        frame,
                        "不支持2D 106点特征点",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                    continue

                # 提取并显示二值肤色图
                skin_mask = self.get_binary_skin_mask(frame, bbox, landmarks_2d)
                if skin_mask is not None:
                    binary_skin_window = skin_mask  # 更新二值图窗口

                # 绘制2D 106点轮廓和姿态信息
                contour_points = self.get_face_contour_points(landmarks_2d)
                frame = self.draw_2d_face_contour(frame, contour_points)

                # 转换为ROS图像消息并发布
                result_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                self.result_pub.publish(result_msg)

                # 发布皮肤掩码
                skin_mask_msg = self.bridge.cv2_to_imgmsg(
                    binary_skin_window, encoding="mono8"
                )
                self.skin_mask_pub.publish(skin_mask_msg)

                pitch, yaw, roll = face["pose"]
                transform_msg = TransformStamped()
                transform_msg.header.stamp = (
                    self.get_clock().now().to_msg()
                )  # 时间戳（ROS 2消息格式）
                transform_msg.header.frame_id = "world"  # 父坐标系
                transform_msg.child_frame_id = "face"  # 子坐标系（如激光雷达）
                transform_msg.transform.translation.x = 0.0
                transform_msg.transform.translation.y = 0.0
                transform_msg.transform.translation.z = 0.0
                transform_msg.transform.rotation = self.euler_to_quaternion(
                    pitch, yaw, roll
                )
                self.tf_broadcaster.sendTransform(transform_msg)

            except Exception as e:
                self.get_logger().error(f"帧处理错误: {str(e)}")

            # 标记处理完成
            with self.lock:
                self.processing = False

    def image_callback(self, msg):
        """图像订阅回调函数"""
        # 如果正在处理中，则只保留最新帧
        with self.lock:
            # 转换ROS图像消息为OpenCV格式
            try:
                self.latest_frame = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
            except Exception as e:
                self.get_logger().error(f"图像转换错误: {str(e)}")
                return


def main(args=None):
    rclpy.init(args=args)

    # 创建节点
    node = FaceDetectionNode()

    # 启动处理线程
    threading.Thread(target=node.process_frame, daemon=True).start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
