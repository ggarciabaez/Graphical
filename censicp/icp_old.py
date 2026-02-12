import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from numpy.linalg import inv, det, norm, solve
import numpy as np


def compute_residuals(p, q1s, normals):
    """
    Compute point-to-line distance residuals.

    Args:
        p: (N, 2) points
        q1s: (N, 2) segment start points
        normals: (N, 2) unit normals

    Returns:
        (N,) absolute distances from p to lines
    """
    return np.abs(np.sum((p - q1s) * normals, axis=1))


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


fig, ax = plt.subplots()

def transform_points(points, tx, ty, theta):
    """Transform points by translation (tx, ty) and rotation theta"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta],
                  [sin_theta, cos_theta]])

    # Apply rotation then translation
    transformed = points @ R.T + np.array([tx, ty])
    return transformed


def angle2rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def perp(v):
    return np.array([-v[1], v[0]])

def project_to_segment(p, q1, q2):
    v = q2 - q1
    t = np.dot(p - q1, v) / np.dot(v, v)
    t = np.clip(t, 0.0, 1.0)
    return q1 + t * v

def project_to_line(p, q, n):
    return p - n * np.dot(n, p - q)

def compose_se2(tf, delta):
    x, y, th = tf
    dx, dy, dth = delta

    R = angle2rot(th)
    t = R @ np.array([dx, dy])

    return np.array([
        x + t[0],
        y + t[1],
        np.arctan2(np.sin(th + dth), np.cos(th + dth))
    ])

# ---------- Correspondences ----------
def vec_neighbors(points, max_seg_dist=0.1, mask=None):
    if mask is not None:
        points = points[mask]
    tree = KDTree(points)
    # k=2 because k=1 is the point itself
    dists, idxs = tree.query(points, k=2)

    # 1. Filter segments: Only keep pairs where distance is below threshold
    valid_mask = dists[:, 1] < max_seg_dist
    valid_idxs = idxs[valid_mask]

    # Create the segments array: shape (N, 2, dimensionality)
    segments = points[valid_idxs]

    # 2. Build the tree using flattened segment points
    # Each segment (q1, q2) will occupy two consecutive slots in the tree
    flat_segments = segments.reshape(-1, points.shape[1])
    segment_tree = KDTree(flat_segments)

    return points, segments, segment_tree

def estimate_normals_2d(points, neighbors=10):
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.zeros((0, 2)), np.zeros((0,))

    k = int(min(max(neighbors, 2), n_pts))
    tree = KDTree(points)
    _, idxNN_all = tree.query(points, k=k)

    normals = np.zeros_like(points)
    linearity = np.zeros((n_pts,), dtype=np.float32)

    for i, idxNN in enumerate(idxNN_all):
        pts = points[idxNN]
        C = np.cov(pts.T, bias=False)
        eig_vals, eig_vecs = np.linalg.eigh(C)
        order = np.argsort(eig_vals)
        n = eig_vecs[:, order[0]]
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n = n / n_norm
        normals[i] = n
        l1 = eig_vals[order[1]]
        l2 = eig_vals[order[0]]
        linearity[i] = (l1 - l2) / (l1 + 1e-12)

    return normals, linearity


def getNaiveCorrespondence(src, segments, tree):
    # Query the tree for the closest endpoint
    dists, point_indices = tree.query(src, workers=-1)

    # 3. Map point index back to segment index
    # Since each segment has 2 points, we use floor division
    segment_indices = (point_indices // 2).astype(int)

    # Extract q1 and q2 for the chosen segments
    q1s = segments[segment_indices, 0]
    q2s = segments[segment_indices, 1]

    return src, q1s, q2s, get_normal(q1s, q2s)

def get_point_normal_correspondence(src, points, tree, normals, linearity=None, min_linearity=0.0):
    _, point_indices = tree.query(src, workers=-1)
    q = points[point_indices]
    n = normals[point_indices]
    if linearity is not None and min_linearity > 0.0:
        keep = linearity[point_indices] >= min_linearity
        return src[keep], q[keep], n[keep]
    return src, q, n


def trim_correspondences(p, q1s, q2s, norm, max_dist=0.5, use_mad=True, mad_sigma=3.0):
    """Remove worst correspondences based on distance and optional MAD filter."""
    distances = np.array([
        np.abs(np.dot(norm[i], p[i] - q1s[i]))
        for i in range(len(p))
    ])
    mask = distances < max_dist
    if use_mad and distances.size > 0:
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        if mad > 0:
            mask &= np.abs(distances - median) <= mad_sigma * mad

    return p[mask], q1s[mask], q2s[mask], norm[mask]

def get_normal(q1s, q2s):
    seg = q2s - q1s
    n = np.stack([-seg[:, 1], seg[:, 0]], axis=1)

    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-10)


# ---------- ICP Linear Solver ----------
def update_transform(p, q1s, q2s, norm, robust_loss="huber", loss_param=0.1, use_point_normals=False):
    """
    Solves the PLICP optimization using the Lagrange Multiplier method
    described in Censi's paper (Appendix I).
    """

    # Compute residuals for weighting
    residuals = compute_residuals(p, q1s, norm)

    # Compute robust weights
    if robust_loss == 'huber':
        weights = huber_weights(residuals, delta=loss_param)
    elif robust_loss == 'cauchy':
        weights = cauchy_weights(residuals, c=loss_param)
    elif robust_loss == 'tukey':
        weights = tukey_weights(residuals, c=loss_param)
    elif robust_loss == 'fair':
        weights = fair_weights(residuals, c=loss_param)
    else:
        weights = np.ones_like(residuals)

    # 1. Build Matrices M and vector g
    M = np.zeros((4, 4))  # ITS ACTUALLY SYMMETRIC!!!
    g = np.zeros((4, 1))

    # C_i = n_i * n_i.T (Point-to-Line metric) [cite: 367]
    # We can vectorize this construction for speed, but the loop is clearer for matching theory
    for i in range(len(p)):
        p_i = p[i]
        n_i = norm[i]
        w_i = weights[i]
        if use_point_normals:
            q_proj = project_to_line(p_i, q1s[i], n_i)
        else:
            q_proj = project_to_segment(p_i, q1s[i], q2s[i])
        # Matrix M_i [cite: 376]
        # x = [tx, ty, cos, sin]
        M_i = np.array([[1.0, 0.0, p_i[0], -p_i[1]],
                        [0.0, 1.0, p_i[1], p_i[0]]])
        C_i = w_i*np.outer(n_i, n_i)
        # print(M_i, C_i, M_i.T@C_i@M_i, np.dot(p_i, n_i), '\n', sep="\n")
        # Accumulate M and g [cite: 19]
        # g term: -2 * pi.T * Ci * Mi. (Here pi is the projection on line)
        # u = C_i@M_i
        M += M_i.T @ C_i@M_i
        g += (-2.0 * q_proj[None, :] @ C_i@M_i).T

    M *= 2.0
    # M[:2, :2] += np.eye(2)*1e-4
    # if direct:
    #     return (-np.linalg.solve(M, g)).ravel()
    # 2. Partition M into A, B, D [cite: 385]
    # 2M + 2lambda*W. We factor out the 2 later or handle it in coefficients.
    # The paper uses 2M in definitions. Let's stick to M_paper = M_code.
    # Equation 23: x = -(2M + 2lambda*W)^-1 * g
    A = M[:2, :2]
    B = M[:2, 2:]
    D = M[2:, 2:]

    # 3. Compute S and S_A [cite: 390, 396]
    A_inv = inv(A)
    S = D - B.T @ A_inv @ B
    S_det = det(S)
    S_trace = np.trace(S)
    S_A = S_det * inv(S)  # Adjugate of S

    # 4. Compute Polynomial Coefficients for P(lambda) = 0 [cite: 400]
    # LHS Terms
    # 4*lambda^2 * g.T * K2 * g
    K2 = np.block([
        [A_inv @ B @ B.T @ A_inv.T, -A_inv @ B],
        [-(A_inv @ B).T, np.eye(2)]
    ])

    # 4*lambda * g.T * K1 * g
    K1 = np.block([
        [A_inv @ B @ S_A @ B.T @ A_inv.T, -A_inv @ B @ S_A],
        [-(A_inv @ B @ S_A).T, S_A]
    ])

    # Constant * g.T * K0 * g
    K0 = np.block([
        [A_inv @ B @ S_A.T @ S_A @ B.T @ A_inv.T, -A_inv @ B @ S_A.T @ S_A],
        [-(A_inv @ B @ S_A.T @ S_A).T, S_A.T @ S_A]
    ])

    c2_lhs = 4.0 * (g.T @ K2 @ g).item()
    c1_lhs = 4.0 * (g.T @ K1 @ g).item()
    c0_lhs = (g.T @ K0 @ g).item()

    # RHS Terms: [p(lambda)]^2 = [4*lam^2 + 2*trace*lam + det]^2
    # This expands to:
    # 16*lam^4 + 16*tr*lam^3 + (4*tr^2 + 8*det)*lam^2 + 4*tr*det*lam + det^2

    # RHS: [4λ² + 2*trace*λ + det]²
    p4 = -16.0
    p3 = -16.0 * S_trace
    p2 = c2_lhs - (4.0 * S_trace ** 2 + 8.0 * S_det)
    p1 = c1_lhs - (4.0 * S_trace * S_det)
    p0 = c0_lhs - S_det ** 2
    roots = np.roots([p4, p3, p2, p1, p0])

    # We need the largest real root
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        return np.zeros(4)  # Fallback
    lambda_opt = np.max(real_roots)

    # 6. Solve for x using Eq 24 [cite: 363]
    W = np.diag([0.0, 0.0, 1.0, 1.0])
    H = M + 2.0 * lambda_opt * W
    try:
        x = solve(-H, g).ravel()
    except np.linalg.LinAlgError:
        print("[ICP]ERROR: Could not solve for x:", H, g, roots, M)
        x = np.zeros(4)
    return x


def ICP(src, tgt, max_iter=20,
        loss="huber", loss_param=0.1, max_dist=0.5,
        max_seg_dist=np.inf, visualize=False, tgt_mask=None,
        use_mad=True, mad_sigma=3.0, use_point_normals=False,
        normal_neighbors=10, min_linearity=0.0):
    """
    Executes the Iterative Closest Point algorithm using Censi's Point-to-Line metric and naive correspondences.
    :param src: The point cloud we'll be transforming
    :param tgt: The target point cloud to transform towards
    :param max_iter: A limit for iterations
    :param loss: The robust loss function to use. Options are: "huber", "cauchy", "tukey", "fair", ""
    :param loss_param: The threshold to use for the chosen loss function.
    :param max_dist: Maximum allowed distance for a correlation to be considered.
    :param max_seg_dist: Maximum allowed distance between two tgt points to be considered a valid segment.
    :param visualize: If true, a matplotlib window pops up to visualize
    :return: The pose for the shifted point cloud, the
    """
    tgt_points, segments, seg_tree = vec_neighbors(tgt, max_seg_dist, mask=tgt_mask)
    if use_point_normals:
        normals, linearity = estimate_normals_2d(tgt_points, neighbors=normal_neighbors)
        point_tree = KDTree(tgt_points)

    # Initialize global transform matrix (3x3 homogeneous)
    T_global = np.eye(3)

    og_hom = np.hstack([src, np.ones((src.shape[0], 1))])  # Homogeneous coords

    for i in range(max_iter):
        # Apply current global transform
        src_curr = (T_global @ og_hom.T).T[:, :2]

        # Get correspondences
        if use_point_normals:
            p, q1s, norm = get_point_normal_correspondence(
                src_curr, tgt_points, point_tree, normals, linearity, min_linearity
            )
            q2s = q1s
        else:
            p, q1s, q2s, norm = getNaiveCorrespondence(src_curr, segments, seg_tree)
        p, q1s, q2s, norm = trim_correspondences(
            p, q1s, q2s, norm, max_dist, use_mad=use_mad, mad_sigma=mad_sigma
        )

        # Visualization
        if visualize:
            ax.clear()
            ax.scatter(*tgt.T, c='red', label='Target', s=10)
            ax.scatter(*src_curr.T, c='blue', alpha=0.5, label='Source', s=10)

            # Draw correspondences
            for j in range(len(q1s)):
                t = project_to_segment(p[j], q1s[j], q2s[j])
                ax.plot([p[j][0], t[0]], [p[j][1], t[1]],
                        color='green', alpha=0.3, linewidth=0.5)
                # Draw the matched segment
                ax.plot([q1s[j][0], q2s[j][0]], [q1s[j][1], q2s[j][1]],
                        color='orange', linewidth=2)
                # Draw normals (every 10th for clarity)
                if j % 10 == 0:
                    mid = (q1s[j] + q2s[j]) / 2
                    ax.arrow(mid[0], mid[1], norm[j, 0] * 0.05, norm[j, 1] * 0.05,
                             color='red', width=0.005, head_width=0.02)

            ax.set_title(f"Iter {i}")
            ax.axis('equal')
            plt.draw()
            plt.pause(1)

        # Solve for INCREMENTAL transform x
        # This x transforms p to the line.
        # Since p is T_global * og, x is applied on top of T_global.
        x_opt = update_transform(
            p, q1s, q2s, norm, loss, loss_param, use_point_normals=use_point_normals
        )
        if x_opt[2] == x_opt[3] == 0.0:
            print("[ERROR] An error occured with ICP. Please check.")
            return np.zeros((3, 3))
        # Construct Incremental Matrix
        # x = [tx, ty, cos, sin]
        dx, dy, c, s = x_opt
        norm_scale = np.sqrt(c * c + s * s)  # Should be 1.0 due to constraint
        # print(norm_scale)  # Works as a REALLY GOOD measure of success, actually!
        c, s = c / norm_scale, s / norm_scale

        T_inc = np.array([
            [c, -s, dx],
            [s, c, dy],
            [0, 0, 1]
        ])

        # Update Global Transform
        T_global = T_inc @ T_global


        # Check convergence (magnitude of incremental update)
        if np.linalg.norm(T_inc[:2, 2]) < 1e-4 and abs(np.arctan2(s, c)) < 1e-4:
            break

    return T_global


def gen_sines(n_pts, dev, x_shift, rot, trans):
    x = np.linspace(0, 6.28, n_pts)
    y = np.sin(x)  # Pure target
    # Add the noise to the data
    y_shift = np.sin(x + x_shift)
    y_noised = y_shift + np.random.normal(0, dev, n_pts)

    return np.array([x, y]).T, transform_points(np.array([x, y_noised]).T, *trans, rot)

if __name__ == "__main__":
    src, tgt = gen_sines(100, 0.00, 0.0, np.deg2rad(10), [1, 1])
    from time import perf_counter

    a = perf_counter()
    tf = ICP(src, tgt, max_dist=5, visualize=True, use_point_normals=False, use_mad=True)
    det = perf_counter() - a
    print("ICP Transform [x, y, theta]:")
    print(tf)
    print("Speed:", 1/det)
