"""Run scan matching on a data run"""
from collections import deque
from mcap_protobuf.reader import read_protobuf_messages
from censicp.icp import ICP, transform_points
from censicp.utilities import *
import rerun as rr
from msgpack import load
import matplotlib.pyplot as plt

dirname = "./data/sm_run1/full_ctrl_demo"
mcap = read_protobuf_messages(dirname + ".mcap")
cm = plt.get_cmap("turbo")

# BEST KNOWN PARAMETERS
radius = 0.05
rpm_coeff = (np.pi * radius / (60 * 11.838))
steer_coeff = 22 / 45
scan_matching=True

icp_params = {
    "max_iter": 50,
    "loss": "cauchy",  # see apply_weights for best parameters.
    "loss_param": 0.185,  # .2 for old
    "max_dist": 1.0,  # best kept at 1.0, actually!
    "normal_neighbors": 3,  # Seems to be only good at 3.
    "mad_sigma": 3.0,  # Small, .05 changes (3.0 seems to be the ideal)
    "verbose": False
}

odom = np.array([0.0, 0.0, 0.0], dtype=np.float32)
movement_acc = np.array([0.0, 0.0], dtype=np.float32)
path_history = []
prev_scan = None
# NEW: Use a global map instead of scan window
global_map = VoxelMap(voxel_size=0.20)  # 5cm voxels

with open("./data/sm_run1/full_ctrl_demo.msgpack", "rb") as f:
    telem = load(f)

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

def get_new_odom(dt, rpm, steer):
    global odom
    v = rpm * rpm_coeff
    dx = v * np.cos(odom[2])
    dy = v * np.sin(odom[2])
    dtheta = v / 0.324 * np.tan(np.deg2rad(steer * steer_coeff))
    # odom = compose_se2(odom, np.array([dx, dy, dtheta], dtype=np.float32) * dt)
    odom += np.array([dx, dy, dtheta], dtype=np.float32) * dt

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
        # Add first scan to map
        curr = pts[:, :2]
        curr_world = downsample_scan(curr, voxel_size=0.05)
        prev_scan = curr_world
        global_map.add_points(curr_world)
        log_pose()

    dt = (msg.log_time - prev_time).total_seconds()
    prev_time = msg.log_time

    # Odometry prediction
    get_new_odom(dt, curr_telem[0]['rpm'], curr_telem[1])
    rr.set_time("record_time", timestamp=msg.log_time)

    # curr = pts[:, :2]
    curr = downsample_scan(pts[:, :2], voxel_size=0.02)
    # HOLY SHIT WE HAVE A GOOD SCAN MATCH LETS GOOOOOOOOOOOOOOO
    if scan_matching:
        prev_scan_world = prev_scan
        # prev_scan_world = global_map.get_local_map(odom, 8.0)

        # This could skew results since we already added current odometry
        odom_inv = inverse_se2(odom)  # Transform previous scan to current sensor frame
        prev_scan_local = transform_points(prev_scan_world, *odom_inv)

        # print(curr.shape, prev_scan_local.shape)
        result = ICP(curr, prev_scan_local, **icp_params)  # ICP in sensor frame

        if result[2, 2] == 0:
            print(f"[ITER {i}] ICP failed")
            icp_params["verbose"] = True
            result = ICP(curr, prev_scan_local, **icp_params)
            print("Result:", result)
            print("Odom:", odom)
            break

        # Extract LOCAL transform (FIXED rotation extraction!)
        shift = np.array([
            result[0, 2],
            result[1, 2],
            np.arctan2(result[1, 0], result[0, 0])
        ])
        distance = np.linalg.norm(shift[:2])
        movement_acc += [distance, np.abs(shift[2])]
        # print(f"Iter {i}: Odom shift: {shift}, Odom: {odom}")

        # Compose with odometry
        odom = compose_se2(odom, shift)
        # odom += shift

    # Update scan window with corrected scan
    curr_world = transform_points(curr, *odom)
    log_cloud = np.hstack((curr_world, np.zeros((len(curr_world), 1))))
    rr.log("world/pointcloud", rr.Points3D(log_cloud, colors=gen_cmap(log_cloud), radii=0.025))
    added = global_map.add_points(curr_world)
    prev_scan = curr_world.copy()


    # Update global map visualization every 10 frames
    if i % 10 == 0:
        map_points = global_map.get_points()
        rr.log(
            "world/global_map",
            rr.Points3D(
                np.hstack((map_points, np.zeros((len(map_points), 1)))),
                radii=0.02,
            ),
        )

    prev_telem = curr_telem
    log_pose()
