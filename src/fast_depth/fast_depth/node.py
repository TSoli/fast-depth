import os

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image as ImgMsg
from torchvision import transforms


class FastDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("fastdepth_node")

        self.declare_parameter(
            "ckpt",
            "",
            ParameterDescriptor(description="FastDepth model checkpoint file"),
        )
        ckpt = self.get_parameter("ckpt").get_parameter_value().string_value
        self.model = torch.load(ckpt)["model"]
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )

        self.image_sub = self.create_subscription(
            ImgMsg, "camera/image", self.get_depth, 10
        )

        self.depth_pub = self.create_publisher(ImgMsg, "camera/depth", 10)

        self.cv_bridge = CvBridge()

    def get_depth(self, msg: ImgMsg) -> None:
        img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        h, w, _ = img.shape
        input = self.transform(img).unsqueeze(0).cuda()
        depth = self.model(input)
        depth_img = depth.squeeze().detach().cpu().numpy()
        depth_img = cv2.resize(depth_img, (w, h))

        depth_msg = self.cv_bridge.cv2_to_imgmsg(
            depth_img, header=msg.header, encoding="passthrough"
        )
        self.depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FastDepthNode()
    rclpy.spin(node)
    rclpy.shutdown()
