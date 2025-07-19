from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy


'''
    reliability (надёжность передачи):
        ReliabilityPolicy.RELIABLE - гарантированная доставка
        ReliabilityPolicy.BEST_EFFORT - нет гарантии доставки, но меньше нагрузка
    history (глубина истории сообщений):
        HistoryPolicy.KEEP_LAST - хранить только первые depth сообщений
        HistoryPolicy.KEEP_ALL - хранить все сообщения, что может перегружать память
    depth (глубина очереди сообщений):
        Определяет, сколько сообщений хранится, если их обработка задерживается
'''


default_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=10, 
)

image_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=1, 
)

costmap_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=1, 
)

laserscan_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=10, 
)

compass_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=3, 
)

imu_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=10, 
)

encoders_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=10, 
)

gps_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=1, 
)

cmd_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=1, 
)

goal_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=5, 
)

odometry_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=5, 
)

scan_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, 
    history=HistoryPolicy.KEEP_ALL, 
)

bridge_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_ALL, 
)

lidar_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=1, 
)

pose_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, 
    history=HistoryPolicy.KEEP_LAST, 
    depth=10, 
)
