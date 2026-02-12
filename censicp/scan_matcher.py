"""
Scan Matching with PLICP - Real Data Processing

This module handles real-time scan matching using the PLICP algorithm
for mobile robot localization and mapping.
"""

import numpy as np
from .icp import ICP, transform_points
from .utilities import compose_se2, inverse_se2, downsample_scan


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
            key = tuple((p / self.voxel_size).astype(int))
            
            if key not in self.points:
                self.points[key] = []
                n_added += 1
            
            self.points[key].append(p)
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

    def get_local_map(self, center, radius):
        """Get points within radius of center (for faster ICP)."""
        points = self.get_points()
        if len(points) == 0:
            return points
        distances = np.linalg.norm(points - center[:2], axis=1)
        return points[distances < radius]

# ============================================================================
# Scan Matcher
# ============================================================================

class ScanMatcher:
    """
    Manages scan-to-map and scan-to-scan matching.
    
    Maintains both a local buffer of recent scans and a global map.
    """
    
    def __init__(self, global_voxel_size=0.05,
                 **kwargs):
        """
        Args:
            local_buffer_size: number of scans to keep in local buffer
            global_voxel_size: voxel size for global map
        Args for ICP:
            src: (N, 2) source point cloud
            tgt: (M, 2) target point cloud
            max_iter: maximum number of iterations
            loss: robust loss function ("huber", "cauchy", "tukey", "fair", "none")
            loss_param: loss function parameter (threshold or scale)
            max_dist: hard distance threshold for correspondences (meters)
            mad_sigma: MAD threshold multiplier for outlier rejection
            normal_neighbors: k for PCA-based normal estimation
            verbose: print iteration info

        """
        self.prev_scan=None
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
        if self.prev_scan is None:
            self.prev_scan = points_world.copy()
            self.global_map.add_points(points_world)
            self.pose_history.append(self.pose.copy())
            return np.array([0.0, 0.0, 0.0])
        
        # Match to previous scan
        prev_scan = self.prev_scan
        
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


class FusedScanMatcher(ScanMatcher):
    """
    Manages scan-to-map and scan-to-scan matching.

    Maintains both a local buffer of recent scans and a global map.
    """

    def __init__(self, global_voxel_size=0.05, odom_weight=0.9,
                 **kwargs):
        super().__init__(global_voxel_size, **kwargs)
        self.odom_weight = odom_weight

    def add_ackerman(self, rpm, steer, dt, comp=False,
                     L=0.324, kRPM=(np.pi * 0.05 / (60 * 11.838)), kSteer=22/45):
        """
        careful with this one, chief. Adds the wheel odometry into the pose estimate using the Ackerman model.
        :param rpm: RPM measured
        :param steer: Steering angle in degrees.
        :param dt: time between current and last measurement
        :param comp: True if the pose should be composed into SE2, False if it should be added directly
        :param L: The wheelbase of the car
        :param kRPM: Constant to convert from rpm to linear velocity
        :param kSteer: Constant to modify the steering bounds. Default maps from 45 to 22 deg.
        :return: The change in pose on the global frame (uses the current pose estimate)
        """
        v = kRPM*rpm
        dx = v*np.cos(self.pose[2])
        dy = v*np.sin(self.pose[2])
        dtheta = v/L * np.tan(np.deg2rad(steer*kSteer))
        delta_pose = np.array([dx, dy, dtheta], dtype=np.float32)*dt
        self.add_odom(delta_pose, comp)
        return delta_pose

    def add_odom(self, delta_pose, composition=False):
        if composition:
            self.pose = compose_se2(self.pose, delta_pose)
        else:
            self.pose += delta_pose

    def add_scan(self, points, voxel_map_radius=0, downsample_vsize=0):
        """
        Add a new scan and update map.

        Args:
            points: (N, 2) new scan in sensor frame
            voxel_map_radius: Radius to extract nearby points from voxel map. If 0, scan-scan matching is used
            downsample_vsize: Voxel size to use to downsample the input point cloud. If 0, the full cloud is used.

        Returns:
            pose_update: [dx, dy, dtheta] correction from scan matching
        """
        # Transform to world frame
        if downsample_vsize > 0:
            points = downsample_scan(points, downsample_vsize)
        points_world = transform_points(points, *self.pose)

        # For first scan, just add to map
        if self.prev_scan is None:
            self.prev_scan = points_world
            self.global_map.add_points(points_world)
            self.pose_history.append(self.pose.copy())
            return np.array([0.0, 0.0, 0.0])

        if voxel_map_radius > 0:
            # --- NEW: get local map instead of previous scan ---
            target = self.global_map.get_local_map(self.pose, voxel_map_radius)

            if len(target) < 20:
                # Fallback to previous scan if map is too sparse
                target = self.prev_scan
        else:
            # Match to previous scan
            target = self.prev_scan

        # Transform previous scan to current sensor frame
        pose_inv = inverse_se2(self.pose)
        target_local = transform_points(target, *pose_inv)

        # Run ICP
        T = ICP(
            points, target_local,
            **self.icp_args
        )

        # Extract pose update
        pose_update = np.array([
            T[0, 2],
            T[1, 2],
            np.arctan2(T[1, 0], T[0, 0])
        ], dtype=np.float32)

        # Update global pose
        # TODO: right here, officer. This guy extracts the pose from an SE2 to turn it back into an SE2.
        self.pose = compose_se2(self.pose, pose_update*self.odom_weight)
        self.pose_history.append(self.pose.copy())

        # Update representations
        points_world = transform_points(points, *self.pose)
        self.prev_scan = points_world
        self.global_map.add_points(points_world)

        return pose_update  # The shift, not the absolute pose. Nice!


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


if __name__ == "__main__":
    # Demonstrate the scan matcher
    print("Scan Matching Demo")
    print("=" * 60)
    
    # Create matcher
    matcher = ScanMatcher(global_voxel_size=0.05)
    
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
