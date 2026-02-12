from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from mcap_protobuf.reader import read_protobuf_messages
from icp_old import ICP, transform_points
from msgpack import load

dirname = "/home/gg-dev/PycharmProjects/Graphical/data/sm_run1/full_ctrl_demo"
mcap = read_protobuf_messages(dirname + ".mcap")
cm = plt.get_cmap("turbo")

radius = 0.038
rpm_coeff = (np.pi * radius / (60 * 11.838))
steer_coeff = 22 / 45

odom = np.array([0.0, 0.0, 0.0], dtype=np.float32)
path_history = []
scan_window = deque(maxlen=1)

with open("./data/sm_run1/full_ctrl_demo.msgpack", "rb") as f:
    telem = load(f)


def decodepc(msg):
    raw = np.frombuffer(msg.data, dtype=np.float32)
    n_cols = msg.point_stride // 4
    return raw.reshape(-1, n_cols)


def gen_cmap(points):
    x_values = points[:, 0]
    x_min, x_max = x_values.min(), x_values.max()
    if x_max == x_min:
        normalized_x = np.zeros_like(x_values)
    else:
        normalized_x = (x_values - x_min) / (x_max - x_min)
    return cm(normalized_x)


def log_pose():
    global odom, path_history
    path_history.append([odom[0], odom[1], 0.0])
    rr.log("world/trajectory", rr.LineStrips3D([path_history], colors=[255, 255, 0], radii=0.))
    rr.log("world/car_marker", rr.Points3D([odom[0], odom[1], 0.0], colors=[255, 0, 0], radii=0.08))
    """
    rr.log(
        "lidar",
        rr.Transform3D(
            rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=odom[2]),
            translation=[odom[0], odom[1], 0],
        ),
    )
    """


def compose_se2(tf: np.ndarray, delta: np.ndarray) -> np.ndarray:
    x, y, th = tf
    dx, dy, dth = delta
    c, s = np.cos(th), np.sin(th)
    t = np.array([c * dx - s * dy, s * dx + c * dy], dtype=float)
    return np.array(
        [x + t[0], y + t[1], np.arctan2(np.sin(th + dth), np.cos(th + dth))],
        dtype=float,
    )


def inverse_se2(tf: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SE(2) transform."""
    x, y, th = tf
    c, s = np.cos(-th), np.sin(-th)
    inv_t = np.array([c * (-x) - s * (-y), s * (-x) + c * (-y)])
    return np.array([inv_t[0], inv_t[1], -th], dtype=float)


def get_new_odom(dt, rpm, steer):
    global odom
    v = rpm * rpm_coeff
    dx = v * np.cos(odom[2])
    dy = v * np.sin(odom[2])
    dtheta = v / 0.324 * np.tan(np.deg2rad(steer * steer_coeff))
    odom = compose_se2(odom, np.array([dx, dy, dtheta], dtype=np.float32) * dt)


rr.init("scan match")
rr.spawn(memory_limit="50%")
rr.log("origin", rr.Points3D([0, 0, 0]), static=True)

prev_time = 0.0
prev_telem = telem[0]
i = 0

for msg in mcap:
    if msg.topic != "/lidar/points":
        continue

    i += 1
    curr_telem = telem[i]
    if curr_telem[0] is None:
        curr_telem = prev_telem

    pts = decodepc(msg.proto_msg)

    if prev_time == 0.0:
        prev_time = msg.log_time
        # Store first scan in world frame
        curr_world = transform_points(pts[:, :2], *odom)
        scan_window.append(curr_world)
        log_pose()
        continue

    dt = (msg.log_time - prev_time).total_seconds()
    prev_time = msg.log_time

    # Odometry prediction
    get_new_odom(dt, curr_telem[0]['rpm'], curr_telem[1])

    rr.set_time("record_time", timestamp=msg.log_time)

    curr = pts[:, :2]  # Keep in sensor frame

    if len(scan_window) == 0:
        curr_world = transform_points(curr, *odom)
        scan_window.append(curr_world)
        log_pose()
        continue

    prev_scan_world = scan_window[0]

    # Transform previous scan to current sensor frame
    odom_inv = inverse_se2(odom)
    prev_scan_local = transform_points(prev_scan_world, *odom_inv)

    # ICP in sensor frame
    result = ICP(curr, prev_scan_local, loss="huber", loss_param=0.01, max_dist=0.8)

    if result[2, 2] == 0:
        print(f"[ITER {i}] ICP failed")
        result = ICP(curr, prev_scan_local, visualize=True)
        print("Result:", result)
        print("Odom:", odom)
        break

    # Extract LOCAL transform (FIXED rotation extraction!)
    shift = np.array([
        result[0, 2],
        result[1, 2],
        np.arctan2(result[1, 0], result[0, 0])
    ])

    # print(f"Iter {i}: Odom shift: {shift}, Odom: {odom}")

    # Compose with odometry
    odom = compose_se2(odom, shift)

    # Update scan window with corrected scan
    curr_world = transform_points(curr, *odom)
    scan_window.append(curr_world)

    # Visualization
    rr.log(
        "lidar/scan_match",
        rr.Points3D(
            np.hstack((prev_scan_world, np.zeros((len(prev_scan_world), 1)))),
            colors=gen_cmap(prev_scan_world),
            radii=0.025,
        ),
    )

    prev_telem = curr_telem
    log_pose()