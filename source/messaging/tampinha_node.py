import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from irobot_create_msgs.action import NavigateToPosition
from std_msgs.msg import String
import math
import logging
import os

class NavigateToFootprintNode(Node):
    def __init__(self):
        super().__init__('navigate_to_footprint_node')
        
        # Action client pointing to the native Create 3 server
        self.action_client = ActionClient(self, NavigateToPosition, "/navigate_to_position")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

        self.init_subscribers()

    def init_subscribers(self):
        # Listen to the topic bridged by is_ros2_gateway
        self.create_subscription(
            String,
            '/object_footprint',
            self.command_callback,
            10
        )

    def command_callback(self, msg):
        try:
            # Split the string "X Y" or "X Y THETA" into a list
            coords = msg.data.split()
            
            if len(coords) >= 2:
                x = float(coords[0])
                y = float(coords[1])
                
                # If "X Y THETA" is sent, use THETA. If only "X Y", assume 0.0.
                theta = float(coords[2]) if len(coords) > 2 else 0.0 
                
                location = [x, y, theta]
                logging.info(f"Alvo recebido do Espaço Inteligente: x={x}, y={y}, theta={theta}")
                self.send_goal(location)
            else:
                logging.warning(f"Mensagem inválida recebida: '{msg.data}'. Esperado pelo menos 'X Y'.")
                
        except ValueError:
            logging.error(f"Erro ao converter a string '{msg.data}' em números reais.")

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return (qx, qy, qz, qw)

    def send_goal(self, location):
        goal_msg = NavigateToPosition.Goal()
        
        # Ensure the robot turns at the end to match the provided theta
        goal_msg.achieve_goal_heading = True 
        
        # Odom reference since we do not have a global map
        goal_msg.goal_pose.header.frame_id = "odom" 
        goal_msg.goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Define position (X, Y, Z)
        goal_msg.goal_pose.pose.position.x = location[0]
        goal_msg.goal_pose.pose.position.y = location[1]
        goal_msg.goal_pose.pose.position.z = 0.0
        
        # Define orientation (convert theta from Euler to quaternion)
        qx, qy, qz, qw = self.euler_to_quaternion(0.0, 0.0, location[2])
        goal_msg.goal_pose.pose.orientation.x = qx
        goal_msg.goal_pose.pose.orientation.y = qy
        goal_msg.goal_pose.pose.orientation.z = qz
        goal_msg.goal_pose.pose.orientation.w = qw
        
        logging.info("Aguardando o servidor de ação /navigate_to_position...")
        self.action_client.wait_for_server()
        logging.info("Enviando comando para o iRobot Create 3 se mover!")
        
        # Send asynchronously
        self.action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NavigateToFootprintNode()
    logging.info('Nó iniciado. Aguardando coordenadas no tópico /object_footprint...')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
    os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'
    main()