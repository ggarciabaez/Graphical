"""
PLICP (Point-to-Line ICP) - Scan Matching Algorithm

Implementation of the PLICP algorithm from Censi (2007):
"An ICP variant using a point-to-line metric"

Uses Censi's closed-form solution for the point-to-line metric with
exact Lagrange multiplier optimization (Appendix I).
"""

import numpy as np
from scipy.spatial import cKDTree as KDTree
from numpy.linalg import inv, det, solve


# ============================================================================
# SE(2) Transformation Utilities
# ============================================================================

def angle2rot(theta):
    """Convert angle to 2x2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_points(points, tx, ty, theta):
    """
    Transform 2D points by rotation and translation.
    
    Args:
        points: (N, 2) array of points
        tx, ty: translation components
        theta: rotation angle in radians
        
    Returns:
        (N, 2) transformed points
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]], dtype=np.float32)
    
    return points @ R.T + np.array([tx, ty], dtype=np.float32)


# ============================================================================
# Geometry Utilities
# ============================================================================

def project_to_line(p, q, n):
    """
    Project point p onto infinite line through q with normal n.
    Same as projecting to a segment without clipping the interpolation (better!)
    Args:
        p: (2,) point
        q: (2,) point on line
        n: (2,) unit normal to line

    Returns:
        (2,) projection of p onto line
    """
    return p - n * np.dot(n, p - q)


# ============================================================================
# Normal Estimation
# ============================================================================

def estimate_normals_pca(points, k=10):  # TODO: SPEED THIS UPPPPPP
    """
    Estimate surface normals using PCA over k-nearest neighbors.

    Args:
        points: (N, 2) point cloud
        k: number of neighbors for PCA
        
    Returns:
        normals: (N, 2) unit normal vectors
    """
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    
    k = int(min(max(k, 2), n_pts))
    tree = KDTree(points)
    _, neighbor_indices = tree.query(points, k=k)
    
    normals = np.zeros((n_pts, 2), dtype=np.float32)

    for i, neighbors in enumerate(neighbor_indices):
        pts = points[neighbors]
        C = np.cov(pts.T, bias=False)
        eig_vals, eig_vecs = np.linalg.eigh(C)
        
        # Normal is eigenvector with smallest eigenvalue
        if np.argmin(eig_vals) != 0:
            print("Found a non-zero min eigenvalue!")
        n = eig_vecs[:, np.argmin(eig_vals)]
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-10:
            n = n / n_norm
        normals[i] = n

    return normals


# ============================================================================
# Correspondence Search
# ============================================================================

def find_point_correspondences(src, tgt_points, tgt_normals):
    """
    Find closest point correspondence with optional linearity filtering.
    
    Args:
        src: (N, 2) source points
        tgt_points: (M, 2) target points
        tgt_normals: (M, 2) target point normals
        
    Returns:
        src_kept: source points (potentially filtered)
        q: matched target points
        normals: normals at matched points
    """
    tree = KDTree(tgt_points)
    _, indices = tree.query(src, workers=-1)
    
    q = tgt_points[indices]
    normals = tgt_normals[indices]

    return src, q, normals


# ============================================================================
# Correspondence Trimming / Outlier Rejection
# ============================================================================

def compute_residuals(p, q1s, normals):
    """
    Compute point-to-line distance residuals.
    
    Args:
        p: (N, 2) points
        q1s: (N, 2) segment start points
        q2s: (N, 2) segment end points
        normals: (N, 2) unit normals
        
    Returns:
        (N,) absolute distances from p to lines
    """
    return np.abs(np.sum((p - q1s) * normals, axis=1))


def trim_correspondences(p, q1s, normals, max_dist=0.5, mad_sigma=3.0):
    """
    Remove outlier correspondences using distance and MAD filtering.
    
    MAD = Median Absolute Deviation. A robust alternative to std dev.
    
    Args:
        p, q1s, q2s, normals: correspondence data
        max_dist: hard distance threshold
        use_mad: if True, also apply MAD-based filtering
        mad_sigma: number of MAD standard deviations for MAD filter
        
    Returns:
        p, q1s, q2s, normals: filtered correspondences
    """
    residuals = compute_residuals(p, q1s, normals)
    mask = residuals < max_dist
    
    if residuals.size > 0:
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        if mad > 1e-10:
            mask &= np.abs(residuals - median) <= mad_sigma * mad
    
    return p[mask], q1s[mask], normals[mask]


# ============================================================================
# Robust Loss Weights
# ============================================================================

def huber_weights(residuals, delta=0.1):
    """
    Huber loss: smooth for small errors, linear for large.
    
    Good general-purpose robust weight function.
    delta=0.05: aggressive outlier rejection
    delta=0.1: moderate
    delta=0.2: lenient
    """
    weights = np.ones_like(residuals, dtype=np.float32)
    outliers = residuals > delta
    weights[outliers] = delta / residuals[outliers]
    return weights


def cauchy_weights(residuals, c=0.1):
    """Cauchy loss: more aggressive than Huber at rejecting outliers."""
    return 1.0 / (1.0 + (residuals / c) ** 2)


def tukey_weights(residuals, c=0.15):
    """Tukey biweight: completely rejects outliers beyond threshold."""
    weights = np.zeros_like(residuals, dtype=np.float32)
    inliers = residuals <= c
    r = residuals[inliers]
    weights[inliers] = (1 - (r / c) ** 2) ** 2
    return weights


def fair_weights(residuals, c=0.1):
    """Fair loss: smooth Cauchy alternative."""
    return 1.0 / (1.0 + np.abs(residuals) / c)


def apply_loss_weights(residuals, loss_type="huber", loss_param=0.1):
    """
    Compute correspondence weights based on residuals and loss function.
    KNOWN GOOD WEIGHTS:
    Huber>=0.2 (really soft, degrades below .2 but still works)
    Cauchy>=0.185 (Best!),
    Tukey>=0.185,
    Fair>=0.085 (Second best)
    Args:
        residuals: (N,) point-to-line distances
        loss_type: "huber", "cauchy", "tukey", "fair", or "none"
        loss_param: scale parameter for loss function
        
    Returns:
        (N,) weights in [0, 1]
    """
    if loss_type == "huber":
        return huber_weights(residuals, delta=loss_param)
    elif loss_type == "cauchy":
        return cauchy_weights(residuals, c=loss_param)
    elif loss_type == "tukey":
        return tukey_weights(residuals, c=loss_param)
    elif loss_type == "fair":
        return fair_weights(residuals, c=loss_param)
    else:
        return np.ones_like(residuals, dtype=np.float32)


# ============================================================================
# PLICP Solver (Censi's Closed-Form Solution)
# ============================================================================

def solve_plicp(p, q1s, normals, loss_type="huber", loss_param=0.1):
    """
    Solve for incremental SE(2) transformation using Censi's method.
    
    Minimizes: sum_i w_i * (n_i^T * (R(theta)*p_i + t - proj_i))^2
    
    Subject to: cos^2(theta) + sin^2(theta) = 1
    
    Uses Lagrange multipliers with a quartic polynomial solution.
    See Censi (2007) Appendix I.
    
    Args:
        p: (N, 2) source points
        q1s: (N, 2) target segment startpoint
        normals: (N, 2) unit normals
        loss_type: robust loss function type
        loss_param: loss function parameter
        
    Returns:
        x: [tx, ty, cos(theta), sin(theta)] SE(2) transformation
    """
    if len(p) == 0:
        return np.zeros(4, dtype=np.float32)
    
    # Compute residuals and weights
    residuals = compute_residuals(p, q1s, normals)
    weights = apply_loss_weights(residuals, loss_type, loss_param)
    
    # Build the system matrices M and g
    # x = [tx, ty, cos, sin] in 4D
    M = np.zeros((4, 4), dtype=np.float32)
    g = np.zeros((4, 1), dtype=np.float32)
    
    for i in range(len(p)):
        p_i = p[i]
        n_i = normals[i]
        w_i = weights[i]
        
        # Project point to target
        q_proj = project_to_line(p_i, q1s[i], n_i)
        
        # Construct M_i matrix (Censi Eq. 17)
        M_i = np.array([[1.0, 0.0, p_i[0], -p_i[1]],
                        [0.0, 1.0, p_i[1], p_i[0]]], dtype=np.float32)
        
        # Weight matrix C_i = w_i * n_i * n_i^T
        C_i = w_i * np.outer(n_i, n_i)
        
        # Accumulate (Censi Eq. 19)
        M += M_i.T @ C_i @ M_i
        g += (-2.0 * q_proj[None, :] @ C_i @ M_i).T
    
    M *= 2.0
    
    # Partition M into blocks (Censi Eq. 26)
    A = M[:2, :2]
    B = M[:2, 2:]
    D = M[2:, 2:]
    
    try:
        A_inv = inv(A)
    except np.linalg.LinAlgError:
        return np.zeros(4, dtype=np.float32)
    
    S = D - B.T @ A_inv @ B
    S_det = det(S)
    S_trace = np.trace(S)
    S_A = S_det * inv(S)  # Adjugate of S
    
    # Build quartic polynomial coefficients
    K2 = np.block([
        [A_inv @ B @ B.T @ A_inv.T, -A_inv @ B],
        [-(A_inv @ B).T, np.eye(2)]
    ])
    
    K1 = np.block([
        [A_inv @ B @ S_A @ B.T @ A_inv.T, -A_inv @ B @ S_A],
        [-(A_inv @ B @ S_A).T, S_A]
    ])
    
    K0 = np.block([
        [A_inv @ B @ S_A.T @ S_A @ B.T @ A_inv.T, -A_inv @ B @ S_A.T @ S_A],
        [-(A_inv @ B @ S_A.T @ S_A).T, S_A.T @ S_A]
    ])
    
    c2_lhs = 4.0 * (g.T @ K2 @ g).item()
    c1_lhs = 4.0 * (g.T @ K1 @ g).item()
    c0_lhs = (g.T @ K0 @ g).item()
    
    # Quartic polynomial: p4*λ^4 + p3*λ^3 + p2*λ^2 + p1*λ + p0 = 0
    p4 = -16.0
    p3 = -16.0 * S_trace
    p2 = c2_lhs - (4.0 * S_trace ** 2 + 8.0 * S_det)
    p1 = c1_lhs - (4.0 * S_trace * S_det)
    p0 = c0_lhs - S_det ** 2
    
    roots = np.roots([p4, p3, p2, p1, p0])
    real_roots = roots[np.isreal(roots)].real
    
    if len(real_roots) == 0:
        return np.zeros(4, dtype=np.float32)
    
    lambda_opt = np.max(real_roots)
    
    # Solve for x (Censi Eq. 24)
    W = np.diag([0.0, 0.0, 1.0, 1.0])
    H = M + 2.0 * lambda_opt * W
    
    try:
        x = solve(-H, g).ravel()
    except np.linalg.LinAlgError:
        return np.zeros(4, dtype=np.float32)
    
    return x.astype(np.float32)


# ============================================================================
# Main ICP Algorithm
# ============================================================================

def ICP(src, tgt, max_iter=50, loss="cauchy", loss_param=0.185, max_dist=1.0,
        mad_sigma=3.0, normal_neighbors=3, verbose=False):
    """
    PLICP: Iterative Closest Point with point-to-line metric.
    
    Aligns source point cloud to target using Censi's PLICP algorithm.
    
    Args:
        src: (N, 2) source point cloud
        tgt: (M, 2) target point cloud
        max_iter: maximum number of iterations
        loss: robust loss function ("huber", "cauchy", "tukey", "fair", "none")
        loss_param: loss function parameter (threshold or scale)
        max_dist: hard distance threshold for correspondences (meters)
        mad_sigma: MAD threshold multiplier for outlier rejection
        normal_neighbors: k for PCA-based normal estimation
        verbose: print iteration info
        
    Returns:
        T: (3, 3) homogeneous transformation matrix
    """
    src = np.asarray(src, dtype=np.float32)
    tgt = np.asarray(tgt, dtype=np.float32)
    
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        raise ValueError("Empty point clouds")
    
    # Prepare target representation
    tgt_normals = estimate_normals_pca(tgt, k=normal_neighbors)

    # Initialize transformation
    T_global = np.eye(3, dtype=np.float32)
    src_hom = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float32)])
    
    for iteration in range(max_iter):
        # Transform source to current frame
        src_curr = (T_global @ src_hom.T).T[:, :2]
        
        # Find correspondences
        src_curr, q, normals = find_point_correspondences(
            src_curr, tgt, tgt_normals
        )
        q1s = q

        # Trim outliers
        src_curr, q1s, normals = trim_correspondences(
            src_curr, q1s, normals, max_dist, mad_sigma
        )
        
        if len(src_curr) < 3:
            if verbose:
                print(f"[ICP] Iteration {iteration}: Too few correspondences, stopping")
            break
        
        # Solve for incremental transform
        x_opt = solve_plicp(
            src_curr, q1s, normals, loss, loss_param
        )
        
        if x_opt[2] == 0.0 and x_opt[3] == 0.0:
            if verbose:
                print(f"[ICP] Iteration {iteration}: Solver failed")
            break
        
        # Extract and normalize rotation
        dx, dy, c, s = x_opt
        norm_cs = np.sqrt(c * c + s * s)
        c, s = c / (norm_cs + 1e-10), s / (norm_cs + 1e-10)
        
        # Build incremental transform
        T_inc = np.array([
            [c, -s, dx],
            [s, c, dy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Compose with global transform
        T_global = T_inc @ T_global
        
        # Check convergence
        trans_norm = np.linalg.norm(T_inc[:2, 2])
        rot_angle = np.abs(np.arctan2(s, c))
        
        if verbose:
            print(f"[ICP] Iteration {iteration}: trans={trans_norm:.6f}, "
                  f"rot={np.degrees(rot_angle):.6f}°, corr={len(src_curr)}")
        
        if trans_norm < 1e-5 and rot_angle < 1e-5:
            if verbose:
                print(f"[ICP] Converged after {iteration + 1} iterations")
            break
    
    return T_global


# ============================================================================
# Utility: Test Data Generation
# ============================================================================

def generate_test_data(n_pts=100, noise_std=0.0, phase_shift=0.0, 
                       rotation=0.0, translation=(0, 0)):
    """
    Generate synthetic sine-wave test data.
    
    Useful for testing and debugging ICP.
    
    Args:
        n_pts: number of points
        noise_std: Gaussian noise standard deviation
        phase_shift: phase shift for source
        rotation: rotation of source (radians)
        translation: (tx, ty) translation of source
        
    Returns:
        src, tgt: two point clouds as (N, 2) arrays
    """
    x = np.linspace(0, 2*np.pi, n_pts)
    y = np.sin(x)
    tgt = np.stack([x, y], axis=1)
    
    y_src = np.sin(x + phase_shift)
    if noise_std > 0:
        y_src += np.random.normal(0, noise_std, n_pts)
    src = np.stack([x, y_src], axis=1)
    src = transform_points(src, *translation, rotation)
    
    return src, tgt


if __name__ == "__main__":
    # Quick test
    src, tgt = generate_test_data(n_pts=100, noise_std=0.1,
                                  rotation=np.deg2rad(10), translation=(1, 1))
    
    T = ICP(src, tgt, verbose=True)
    
    print("\nFinal transformation matrix:")
    print(T)
    print(f"\nExtracted pose: x={T[0,2]:.4f}, y={T[1,2]:.4f}, "
          f"theta={np.degrees(np.arctan2(T[1,0], T[0,0])):.4f}°")

    from matplotlib import pyplot as plt
    plt.scatter(*src.T, label="src")
    plt.scatter(*tgt.T, label="tgt")
    transformed_points = transform_points(src, T[0, 2], T[1, 2], np.arctan2(T[1,0], T[0,0]))
    plt.scatter(*transformed_points.T, label="transformed_points")  # transforms are weird, man.
    plt.legend()
    plt.show()