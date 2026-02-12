"""
Run scan matching using FusedScanMatcher
"""

from mcap_protobuf.reader import read_protobuf_messages
from censicp.utilities import *
from censicp.icp import transform_points
from censicp.scan_matcher import FusedScanMatcher
from msgpack import load
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt

dirname = "./data/sm_run1/full_ctrl_demo"
mcap = read_protobuf_messages(dirname + ".mcap")

with open("./data/sm_run1/full_ctrl_demo.msgpack", "rb") as f:
    telem = load(f)

cm = plt.get_cmap("turbo")

# ICP parameters (same as before)
icp_params = {
    "max_iter": 50,
    "loss": "cauchy",
    "loss_param": 0.185,
    "max_dist": 1.0,
    "normal_neighbors": 3,
    "mad_sigma": 3.0,
    "verbose": False
}

# Initialize matcher
matcher = FusedScanMatcher(
    global_voxel_size=0.1,
    odom_weight=1.0,        # match old behavior exactly
    **icp_params
)

rr.init("scan match")
rr.spawn(memory_limit="50%")
rr.log("origin", rr.Points3D([0, 0, 0]), static=True)

def gen_cmap(points):
    x_values = points[:, 0]
    x_min, x_max = x_values.min(), x_values.max()
    if x_max == x_min:
        normalized_x = np.zeros_like(x_values)
    else:
        normalized_x = (x_values - x_min) / (x_max - x_min)
    return cm(normalized_x)

prev_time = None
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

    if prev_time is None:
        prev_time = msg.log_time
        matcher.add_scan(pts[:, :2])  # add the first scan :)

    dt = (msg.log_time - prev_time).total_seconds()
    prev_time = msg.log_time

    rr.set_time("record_time", timestamp=msg.log_time)

    # --- 1️⃣ Add wheel odometry ---
    matcher.add_ackerman(
        rpm=curr_telem[0]['rpm'],
        steer=curr_telem[1],
        dt=dt,
        comp=False   # IMPORTANT: matches old compose_se2 behavior
    )

    # --- 3️⃣ Run scan matching ---
    pose_update = matcher.add_scan(pts[:, :2], 0.0, 0.02)

    # --- 4️⃣ Visualization ---
    pose = matcher.get_pose()
    map_points = matcher.get_map()
    trajectory = matcher.get_trajectory()

    # Log trajectory
    rr.log(
        "world/trajectory",
        rr.LineStrips3D([np.hstack((trajectory[:, :2],
                                    np.zeros((len(trajectory), 1))))],
                       colors=[255, 255, 0])
    )

    # Log vehicle marker
    rr.log(
        "world/car_marker",
        rr.Points3D([[pose[0], pose[1], 0.0]],
                    colors=[255, 0, 0],
                    radii=0.08)
    )

    # Log point cloud
    curr_world = transform_points(pts[:, :2], *pose)
    log_cloud = np.hstack((curr_world, np.zeros((len(curr_world), 1))))
    rr.log(
        "world/pointcloud",
        rr.Points3D(log_cloud,
                    colors=gen_cmap(log_cloud),
                    radii=0.025)
    )

    # Log global map every 10 frames
    if i % 10 == 0:
        rr.log(
            "world/global_map",
            rr.Points3D(
                np.hstack((map_points,
                           np.zeros((len(map_points), 1)))),
                radii=0.02,
            )
        )

    prev_telem = curr_telem
