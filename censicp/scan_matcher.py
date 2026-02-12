"""
Scan Matching with PLICP - Real Data Processing

This module handles real-time scan matching using the PLICP algorithm
for mobile robot localization and mapping.
"""

import numpy as np
from collections import deque
from icp import ICP, transform_points, angle2rot


# ============================================================================
# SE(2) Utilities
# ============================================================================

def inverse_se2(pose):
    """
    Compute inverse of SE(2) transformation.
    
    Args:
        pose: [x, y, theta]
        
    Returns:
        [x_inv, y_inv, theta_inv]
    """
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        -(x * c + y * s),
        -(-x * s + y * c),
        -theta
    ])


def compose_se2(tf, delta):
    """
    Compose two SE(2) transformations.
    
    Args:
        tf: [x, y, theta] first pose
        delta: [dx, dy, dtheta] increment in local frame
        
    Returns:
        [x', y', theta'] composed pose
    """
    x, y, th = tf
    dx, dy, dth = delta
    
    c, s = np.cos(th), np.sin(th)
    return np.array([
        x + c * dx - s * dy,
        y + s * dx + c * dy,
        th + dth
    ])


# ============================================================================
# Voxel Grid Map (for data management)
# ============================================================================

class VoxelMap:
    """Simple voxel grid for downsampling and deduplication."""
    
    def __init__(self, voxel_size=0.1):
        """
        Args:
            voxel_size: size of voxels in meters
        """
        self.voxel_size = voxel_size
        self.points = {}  # Dict mapping voxel coords to point lists
    
    def add_points(self, points):
        """
        Add points to the map.
        
        Args:
            points: (N, 2) array of points
            
        Returns:
            n_added: number of new voxels populated
        """
        n_added = 0
        
        for p in points:
            voxel = tuple(np.floor(p / self.voxel_size).astype(int))
            
            if voxel not in self.points:
                self.points[voxel] = []
                n_added += 1
            
            self.points[voxel].append(p)
        
        return n_added
    
    def get_points(self):
        """
        Get all points in the map.
        
        Returns:
            (N, 2) array of points
        """
        all_points = []
        for plist in self.points.values():
            all_points.extend(plist)
        
        if all_points:
            return np.array(all_points)
        else:
            return np.zeros((0, 2))
    
    def get_voxel_centers(self):
        """Get center point of each voxel."""
        centers = []
        for voxel, plist in self.points.items():
            centers.append(np.mean(plist, axis=0))
        return np.array(centers)
    
    def clear(self):
        """Clear all points."""
        self.points.clear()
    
    def size(self):
        """Number of occupied voxels."""
        return len(self.points)


# ============================================================================
# Point Cloud Processing
# ============================================================================

def downsample_scan(points, voxel_size=0.05):
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: (N, 2) point cloud
        voxel_size: voxel size in meters
        
    Returns:
        (M, 2) downsampled points
    """
    voxel_map = VoxelMap(voxel_size)
    voxel_map.add_points(points)
    return voxel_map.get_voxel_centers()


def filter_range(points, max_range=10.0):
    """
    Filter points beyond max range (useful for LiDAR).
    
    Args:
        points: (N, 2) point cloud
        max_range: maximum range in meters
        
    Returns:
        filtered: (M, 2) points within range
    """
    distances = np.linalg.norm(points, axis=1)
    mask = distances < max_range
    return points[mask]


def filter_angle_sector(points, angles, min_angle=None, max_angle=None):
    """
    Filter points by angular sector.
    
    Args:
        points: (N, 2) point cloud
        angles: (N,) angle for each point (radians)
        min_angle, max_angle: angular bounds (radians)
        
    Returns:
        filtered: (M, 2) points in sector
    """
    mask = np.ones(len(points), dtype=bool)
    
    if min_angle is not None:
        mask &= angles >= min_angle
    if max_angle is not None:
        mask &= angles <= max_angle
    
    return points[mask]


# ============================================================================
# Scan Matcher
# ============================================================================

class ScanMatcher:
    """
    Manages scan-to-map and scan-to-scan matching.
    
    Maintains both a local buffer of recent scans and a global map.
    """
    
    def __init__(self, local_buffer_size=5, global_voxel_size=0.05,
                 **kwargs):
        """
        Args:
            local_buffer_size: number of scans to keep in local buffer
            global_voxel_size: voxel size for global map
            icp_max_dist: max correspondence distance for ICP
            icp_loss: loss function for robust matching
        Args for ICP:
            max_iter: maximum number of iterations
            loss: robust loss function ("huber", "cauchy", "tukey", "fair", "none")
            loss_param: loss function parameter (threshold or scale)
            max_dist: hard distance threshold for correspondences (meters)
            max_seg_dist: max distance between points to form segment (meters)
            use_mad: use MAD-based outlier filtering
            mad_sigma: MAD threshold multiplier for outlier rejection
            use_point_normals: if True, use estimated normals instead of segments
            normal_neighbors: k for PCA-based normal estimation
            min_linearity: minimum linearity score to accept correspondence
            verbose: print iteration info
        """
        self.scan_buffer = deque(maxlen=local_buffer_size)
        self.global_map = VoxelMap(global_voxel_size)
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pose_history = []
        self.icp_args = kwargs
    
    def add_scan(self, points):
        """
        Add a new scan and update map.
        
        Args:
            points: (N, 2) new scan in sensor frame
            
        Returns:
            pose_update: [dx, dy, dtheta] correction from scan matching
        """
        # Transform to world frame
        points_world = transform_points(points, *self.pose)
        
        # For first scan, just add to map
        if len(self.scan_buffer) == 0:
            self.scan_buffer.append(points_world)
            self.global_map.add_points(points_world)
            self.pose_history.append(self.pose.copy())
            return np.array([0.0, 0.0, 0.0])
        
        # Match to previous scan
        prev_scan = self.scan_buffer[-1]
        
        # Transform previous scan to current sensor frame
        pose_inv = inverse_se2(self.pose)
        prev_scan_local = transform_points(prev_scan, *pose_inv)
        
        # Run ICP
        T = ICP(
            points, prev_scan_local,
            **self.icp_args
        )
        
        # Extract pose update
        pose_update = np.array([
            T[0, 2],
            T[1, 2],
            np.arctan2(T[1, 0], T[0, 0])
        ], dtype=np.float32)
        
        # Update global pose
        self.pose = compose_se2(self.pose, pose_update)
        self.pose_history.append(self.pose.copy())
        
        # Update representations
        points_world = transform_points(points, *self.pose)
        self.scan_buffer.append(points_world)
        self.global_map.add_points(points_world)
        
        return pose_update
    
    def get_map(self):
        """Get current global map."""
        return self.global_map.get_points()
    
    def get_pose(self):
        """Get current pose estimate."""
        return self.pose.copy()
    
    def get_trajectory(self):
        """Get full pose history."""
        return np.array(self.pose_history)
    
    def reset(self):
        """Reset matcher."""
        self.scan_buffer.clear()
        self.global_map.clear()
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pose_history.clear()


# ============================================================================
# Odometry Correction
# ============================================================================

class OdometryCorrector:
    """
    Fuses odometry with scan-matching corrections.
    
    Allows incremental odometry estimates to be corrected by periodic
    scan matching.
    """
    
    def __init__(self, correction_weight=0.5):
        """
        Args:
            correction_weight: how much to trust scan matching vs odometry
                            0 = pure odometry, 1 = pure scan matching
        """
        self.pose = np.array([0.0, 0.0, 0.0])
        self.correction_weight = correction_weight
    
    def update_odometry(self, delta_pose, composition=False):
        """
        Apply odometry increment. Assumes that dt is already applied.
        
        Args:
            delta_pose: [dx, dy, dtheta] in robot frame
        """
        if composition:
            self.pose = compose_se2(self.pose, delta_pose)
        else:
            self.pose += delta_pose
    
    def apply_correction(self, scan_correction):  #
        """
        Apply scan-matching correction.
        This seems odd to me. I don't trust it too much.
        
        Args:
            scan_correction: [dx, dy, dtheta] correction from scan matching
        """
        # Blend between odometry and scan matching
        corrected = compose_se2(self.pose, scan_correction * self.correction_weight)
        self.pose = corrected
    
    def get_pose(self):
        """Get current estimate."""
        return self.pose.copy()


# ============================================================================
# Example Usage (if running as main)
# ============================================================================

if __name__ == "__main__":
    # Demonstrate the scan matcher
    print("Scan Matching Demo")
    print("=" * 60)
    
    # Create matcher
    matcher = ScanMatcher(local_buffer_size=5, global_voxel_size=0.05)
    
    # Generate fake scans
    np.random.seed(42)
    
    for i in range(10):
        # Simulate scan from robot moving in circle
        angle = 2 * np.pi * i / 10
        radius = 2.0
        true_x = radius * np.cos(angle)
        true_y = radius * np.sin(angle)
        true_theta = angle
        
        # Generate scan (points on a wall)
        x_wall = np.linspace(0, 1, 50)
        y_wall = np.ones(50) * 0.5
        scan = np.stack([x_wall, y_wall], axis=1)
        
        # Add some noise
        scan += np.random.normal(0, 0.01, scan.shape)
        
        # Process scan
        pose_update = matcher.add_scan(scan)
        pose = matcher.get_pose()
        
        print(f"Scan {i:2d}: pose=[{pose[0]:6.3f}, {pose[1]:6.3f}, "
              f"{np.degrees(pose[2]):7.2f}°], "
              f"update=[{pose_update[0]:6.3f}, {pose_update[1]:6.3f}, "
              f"{np.degrees(pose_update[2]):6.2f}°]")
    
    # Print final map size
    print(f"\nFinal map: {matcher.global_map.size()} voxels, "
          f"{len(matcher.get_map())} points")
