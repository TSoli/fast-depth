import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image as ImgMsg


class Viewer(Node):
    def __init__(self):
        super().__init__("viewer")
        self.subscription = self.create_subscription(
            ImgMsg, "camera/depth", self.show_depth, 10
        )
        self.cv_bridge = CvBridge()

    def show_depth(self, msg: ImgMsg) -> None:
        img = self.cv_bridge.imgmsg_to_cv2(msg)
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        cv2.imshow("frame", img)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    viewer = Viewer()
    rclpy.spin(viewer)
    viewer.destroy_node()
    rclpy.shutdown()
