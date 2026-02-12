import numpy as np

class OldVoxelMap:
    """Efficient voxel grid for storing unique points."""

    def __init__(self, voxel_size=0.05):
        self.voxel_size = float(voxel_size)
        self.inv_voxel_size = 1.0 / self.voxel_size
        self.voxels = {}  # Dict of voxel_key -> point

    def _get_voxel_key(self, point):
        """Convert point to voxel grid coordinates."""
        key = np.floor(point * self.inv_voxel_size).astype(np.int32)
        return int(key[0]), int(key[1])

    def add_points(self, points):
        """Add points to the map, deduplicating via voxel grid."""
        points = np.asarray(points, dtype=np.float32)
        added_count = 0
        for point in points:
            key = self._get_voxel_key(point)
            if key not in self.voxels:
                self.voxels[key] = point
                added_count += 1
        return added_count

    def get_points(self):
        """Get all unique points in the map."""
        if not self.voxels:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(list(self.voxels.values()), dtype=np.float32)

    def get_local_map(self, center, radius):
        """Get points within radius of center (for faster ICP)."""
        if not self.voxels:
            return np.zeros((0, 2), dtype=np.float32)

        center_xy = np.asarray(center[:2], dtype=np.float32)
        center_key = np.floor(center_xy * self.inv_voxel_size).astype(np.int32)
        vr = int(np.ceil(radius * self.inv_voxel_size))
        radius2 = float(radius * radius)
        out = []

        for ix in range(int(center_key[0]) - vr, int(center_key[0]) + vr + 1):
            for iy in range(int(center_key[1]) - vr, int(center_key[1]) + vr + 1):
                point = self.voxels.get((ix, iy))
                if point is None:
                    continue
                dx = point[0] - center_xy[0]
                dy = point[1] - center_xy[1]
                if dx * dx + dy * dy <= radius2:
                    out.append(point)

        if not out:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def size(self):
        return len(self.voxels)


class VoxelMap:
    """Simple voxel grid for downsampling and deduplication."""

    def __init__(self, voxel_size=0.1):
        """
        Args:
            voxel_size: size of voxels in meters
        """
        self.voxel_size = float(voxel_size)
        self.inv_voxel_size = 1.0 / self.voxel_size
        self.points = {}  # Dict[(ix, iy)] -> [mean_x, mean_y, count]
        self._points_cache = np.zeros((0, 2), dtype=np.float32)
        self._dirty_cache = False

    def add_points(self, points):
        """
        Add points to the map.

        Args:
            points: (N, 2) array of points

        Returns:
            n_added: number of new voxels populated
        """
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return 0

        keys = np.floor(points * self.inv_voxel_size).astype(np.int32)
        n_added = 0

        for p, key_arr in zip(points, keys):
            key = (int(key_arr[0]), int(key_arr[1]))
            entry = self.points.get(key)
            if entry is None:
                self.points[key] = np.array([p[0], p[1], 1.0], dtype=np.float32)
                n_added += 1
            """
            else:
                count = entry[2] + 1.0
                alpha = 1.0 / count
                entry[0] += (p[0] - entry[0]) * alpha
                entry[1] += (p[1] - entry[1]) * alpha
                entry[2] = count
            """

        self._dirty_cache = True
        return n_added

    def get_points(self):
        """
        Get all points in the map.

        Returns:
            (N, 2) array of points
        """
        if not self.points:
            return np.zeros((0, 2), dtype=np.float32)
        if self._dirty_cache:
            vals = np.fromiter(
                (v[0] for v in self.points.values()),
                dtype=np.float32,
                count=len(self.points),
            )
            vys = np.fromiter(
                (v[1] for v in self.points.values()),
                dtype=np.float32,
                count=len(self.points),
            )
            self._points_cache = np.column_stack((vals, vys))
            self._dirty_cache = False
        return self._points_cache

    def get_voxel_centers(self):
        """Get center point of each voxel."""
        return self.get_points()

    def clear(self):
        """Clear all points."""
        self.points.clear()
        self._points_cache = np.zeros((0, 2), dtype=np.float32)
        self._dirty_cache = False

    def size(self):
        """Number of occupied voxels."""
        return len(self.points)

    def get_local_map(self, center, radius):
        """Get points within radius of center (for faster ICP)."""
        if not self.points:
            return np.zeros((0, 2), dtype=np.float32)

        center_xy = np.asarray(center[:2], dtype=np.float32)
        center_key = np.floor(center_xy * self.inv_voxel_size).astype(np.int32)
        vr = int(np.ceil(radius * self.inv_voxel_size))
        radius2 = float(radius * radius)

        out = []
        for ix in range(int(center_key[0]) - vr, int(center_key[0]) + vr + 1):
            for iy in range(int(center_key[1]) - vr, int(center_key[1]) + vr + 1):
                entry = self.points.get((ix, iy))
                if entry is None:
                    continue
                dx = entry[0] - center_xy[0]
                dy = entry[1] - center_xy[1]
                if dx * dx + dy * dy <= radius2:
                    out.append([entry[0], entry[1]])

        if not out:
            return np.zeros((0, 2), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)


# Add this downsampling function
# TODO: this seems a bit heavy, no?
def downsample_scan(points, voxel_size=0.05):
    """Quick voxel downsampling for a single scan."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.reshape(0, 2)

    inv_voxel = 1.0 / float(voxel_size)
    # voxel_keys = np.floor(points * inv_voxel).astype(np.int32)
    voxel_keys = (points * inv_voxel).astype(np.int32)  # Bad old behavior
    _, unique_idx = np.unique(voxel_keys, axis=0, return_index=True)
    return points[unique_idx]

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