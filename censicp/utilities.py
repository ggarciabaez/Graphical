import numpy as np

# Add this class at the top of your file
class VoxelMap:
    """Efficient voxel grid for storing unique points."""

    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size
        self.voxels = {}  # Dict of voxel_key -> point

    def _get_voxel_key(self, point):
        """Convert point to voxel grid coordinates."""
        return tuple((point / self.voxel_size).astype(int))

    def add_points(self, points):
        """Add points to the map, deduplicating via voxel grid."""
        added_count = 0
        for point in points:
            key = self._get_voxel_key(point)
            if key not in self.voxels:
                self.voxels[key] = point
                added_count += 1
        return added_count

    def get_points(self):
        """Get all unique points in the map."""
        return np.array(list(self.voxels.values()))

    def get_local_map(self, center, radius):
        """Get points within radius of center (for faster ICP)."""
        points = self.get_points()
        if len(points) == 0:
            return points
        distances = np.linalg.norm(points - center[:2], axis=1)
        print(np.mean(points[distances<radius], axis=0))
        return points[distances < radius]

    def size(self):
        return len(self.voxels)


# Add this downsampling function
# TODO: this seems a bit heavy, no?
def downsample_scan(points, voxel_size=0.05):
    """Quick voxel downsampling for a single scan."""
    if len(points) == 0:
        return points

    voxel_keys = {}
    for point in points:
        key = tuple((point / voxel_size).astype(int))
        if key not in voxel_keys:
            voxel_keys[key] = point
    out = np.array(list(voxel_keys.values()))
    print(len(points), len(out))
    return np.array(list(voxel_keys.values()))

def decodepc(msg):
    raw = np.frombuffer(msg.data, dtype=np.float32)
    n_cols = msg.point_stride // 4
    return raw.reshape(-1, n_cols)

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