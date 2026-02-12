"""Test 2 scans from the mcap, in isolation."""

from mcap_protobuf.reader import read_protobuf_messages  # mcap-protobuf-support
from censicp.icp import ICP
import numpy as np
import matplotlib.pyplot as plt
from msgpack import load
dirname = "/home/gg-dev/PycharmProjects/Graphical/data/sm_run1/full_ctrl_demo"
def decodepc(msg):  # expects the proto_msg
    raw = np.frombuffer(msg.data, dtype=np.float32)
    n_cols = msg.point_stride // 4
    return raw.reshape(-1, n_cols)

mcap = read_protobuf_messages(dirname+".mcap")
i = 0
initial = None
curr = None
for n in mcap:
    if n.topic == "/lidar/points":
        i += 1
        if initial is None and i > 45:
            initial = n
        elif i > 50:
            curr = n
            break

src = decodepc(initial.proto_msg)
tgt = decodepc(curr.proto_msg)
vals = ICP(src[:, :2], tgt[:, :2])
print(vals)