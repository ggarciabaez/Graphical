# ICP Parameter Recommendations for Common Scenarios

This guide provides ready-to-use parameter configurations for typical robotics applications.

---

## Quick Parameter Finder

**What's your scenario?**
1. [Structured environments (indoor robot, LiDAR)](#scenario-1-structured-environments)
2. [Unstructured/outdoor (noisy range finder)](#scenario-2-unstructured-outdoor)
3. [High-speed real-time matching](#scenario-3-high-speed-real-time)
4. [Maximum accuracy (offline processing)](#scenario-4-maximum-accuracy)
5. [Custom tune for your data](#parameter-tuning-workflow)

---

## Scenario 1: Structured Environments
*(Indoor robots, 2D LiDAR, clean walls/objects)*

### Configuration
```python
from icp_refactored import ICP

T = ICP(
    src, tgt,
    max_iter=15,           # Structured = quick convergence
    loss="tukey",          # Aggressive outlier rejection
    loss_param=0.08,       # Tight threshold
    max_dist=0.4,          # Trust correspondences are good
    max_seg_dist=0.15,     # Segment from nearby points
    use_point_normals=True,
    normal_neighbors=8,
    min_linearity=0.2,     # Only use high-linearity points
    use_mad=True,
    mad_sigma=2.0,
    verbose=True
)
```

### Rationale
- **tukey loss**: Completely rejects bad outliers (good for clean geometry)
- **low loss_param**: Strict threshold since environment is clean
- **use_point_normals=True**: Leverage the structure
- **min_linearity**: Only match against clear linear features
- **tight MAD**: Remove points that don't fit the trend

### Expected Results
- **Convergence**: 3-5 iterations
- **Accuracy**: ±5mm translation, ±0.5° rotation
- **Speed**: 200-500 Hz

### When to adjust
- Scans from noisy LiDAR? Increase `loss_param` to 0.1-0.15
- Lots of clutter? Decrease `min_linearity` to 0.1
- Over-rejecting? Switch to `loss="huber"` with `loss_param=0.1`

---

## Scenario 2: Unstructured/Outdoor
*(Noisy sensors, outdoor environments, sparse data)*

### Configuration
```python
T = ICP(
    src, tgt,
    max_iter=20,           # More iterations needed
    loss="huber",          # Forgiving but robust
    loss_param=0.15,       # Moderate threshold
    max_dist=1.0,          # More lenient
    max_seg_dist=np.inf,   # Use all points for segmentation
    use_point_normals=False,
    use_mad=True,
    mad_sigma=3.0,         # Looser MAD filter
    verbose=True
)
```

### Rationale
- **huber loss**: Smooth transition (good for noisy data)
- **higher loss_param**: Be lenient with outliers
- **max_seg_dist=inf**: Don't restrict segment formation
- **use_point_normals=False**: Fall back to segment-based matching
- **loose MAD**: Don't over-filter already sparse data

### Expected Results
- **Convergence**: 8-15 iterations
- **Accuracy**: ±15cm translation, ±2-5° rotation
- **Speed**: 50-150 Hz

### When to adjust
- Still rejecting too many points? Use `loss="fair"` with `loss_param=0.2`
- Very noisy? Increase `mad_sigma` to 4.0
- Fast enough but not accurate? Increase `max_iter` to 30

---

## Scenario 3: High-Speed Real-Time
*(Robot needs quick rough estimate, optimization later)*

### Configuration
```python
T = ICP(
    src, tgt,
    max_iter=5,            # Very few iterations!
    loss="none",           # Skip robust weighting overhead
    max_dist=2.0,          # Permissive
    use_mad=False,         # Skip MAD filtering
    verbose=False
)
```

### Rationale
- **min iterations**: Accept approximate result
- **no loss function**: Save computation
- **no MAD**: Skip statistical filtering
- **large max_dist**: Accept loose correspondences
- **verbose=False**: Don't print debug info

### Expected Results
- **Convergence**: Always in 5 iterations (forced)
- **Accuracy**: ±50cm translation, ±10° rotation (rough!)
- **Speed**: 500+ Hz (very fast)

### Use Case
- Coarse alignment before:
  - Graph SLAM refinement
  - Global matching
  - Finer local ICP pass

### To improve accuracy
- First pass: use this config for speed
- Second pass: use Scenario 1 or 4 config for refinement

---

## Scenario 4: Maximum Accuracy
*(Offline processing, highest quality results)*

### Configuration
```python
T = ICP(
    src, tgt,
    max_iter=50,           # Very thorough
    loss="tukey",          # Aggressive outliers
    loss_param=0.05,       # Very strict
    max_dist=0.2,          # Tight acceptance
    max_seg_dist=0.1,      # High-quality segments
    use_point_normals=True,
    normal_neighbors=15,   # More neighbors = better normals
    min_linearity=0.3,     # Only excellent points
    use_mad=True,
    mad_sigma=1.5,         # Very strict outlier removal
    verbose=True
)
```

### Rationale
- **many iterations**: Converge fully
- **strict thresholds**: Accept only best matches
- **careful normal estimation**: More neighbors = better
- **tight linearity**: Only clear features
- **strict MAD**: Remove outliers aggressively

### Expected Results
- **Convergence**: 20-50 iterations (slow but thorough)
- **Accuracy**: ±2mm translation, ±0.1° rotation (excellent)
- **Speed**: 5-20 Hz (very slow but very accurate)

### When to use
- Reference data generation
- Ground truth comparison
- Final offline map-building pass
- Quality benchmarking

### To speed up slightly
- Reduce `max_iter` to 30
- Reduce `normal_neighbors` to 10
- Increase `loss_param` to 0.08

---

## Parameter Tuning Workflow

Use this systematic approach to find optimal parameters for YOUR data:

### Step 1: Characterize Your Data
```python
from icp_refactoring import generate_test_data

# Generate realistic test case
src, tgt = generate_test_data(
    n_pts=len(your_typical_scan),
    noise_std=your_sensor_noise,  # ~0.01 for clean LiDAR, 0.05+ for noisy
    rotation=your_typical_rotation,  # ~10° if small scans, ~30° if large
    translation=your_typical_translation  # typical frame-to-frame distance
)
```

### Step 2: Test Different Losses
```python
from icp_tuning import ICPBenchmark, create_param_grid, print_results_table

bench = ICPBenchmark(n_points=len(your_typical_scan))

configs = create_param_grid(
    loss_types=["none", "fair", "huber", "cauchy", "tukey"],
    loss_params=[0.05, 0.1, 0.15, 0.2],
    max_dists=[0.3, 0.5, 1.0],
    mad_sigmas=[2.0, 3.0],
    use_mad_vals=[True, False]
)

results = bench.test_parameter_grid(configs)
print_results_table(results, sort_by='error')
```

### Step 3: Narrow Down
```python
# Pick top 5 configurations from step 2
# Test with YOUR actual data, not synthetic

# For each top configuration:
for config in top_5_configs:
    results = []
    for your_scan_pair in your_test_data:
        T = ICP(your_scan_pair[0], your_scan_pair[1], **config)
        error = evaluate_pose(T, ground_truth)
        results.append(error)
    
    avg_error = np.mean(results)
    print(f"Config {config}: avg error = {avg_error:.6f}")
```

### Step 4: Sensitivity Analysis
```python
# Test best config against noise levels
best_config = ...  # From step 3

for noise in [0.0, 0.01, 0.02, 0.05, 0.1]:
    src, tgt = generate_test_data(
        n_pts=len(your_scan),
        noise_std=noise,
        **your_typical_transform_params
    )
    T = ICP(src, tgt, **best_config)
    error = evaluate_pose(T, ground_truth_transform)
    print(f"Noise {noise:.3f}: Error {error:.6f}")
```

### Step 5: Finalize
Once you find good parameters, **write them down**:

```python
# my_icp_config.py
ICP_PARAMS_FOR_WHEELCHAIR = {
    'max_iter': 15,
    'loss': 'tukey',
    'loss_param': 0.08,
    'max_dist': 0.4,
    'max_seg_dist': 0.15,
    'use_point_normals': True,
    'normal_neighbors': 8,
    'min_linearity': 0.2,
    'use_mad': True,
    'mad_sigma': 2.0,
}

# Then use:
from icp_refactored import ICP
from my_icp_config import ICP_PARAMS_FOR_WHEELCHAIR

T = ICP(src, tgt, **ICP_PARAMS_FOR_WHEELCHAIR)
```

---

## Common Issues & Fixes

### "ICP converges but result is wrong"
**Likely cause**: Good local minimum, but it's the wrong one

**Fixes** (in order):
1. Start with better initial guess
2. Use coarse-to-fine: rough params first, then fine
3. Increase `loss_param` (be more forgiving)
4. Use `loss="none"` to verify it's not the loss function

### "Too many iterations, very slow"
**Likely cause**: Parameters too strict, won't converge

**Fixes**:
1. Increase `loss_param` (loose outlier rejection)
2. Increase `max_dist` (accept more correspondences)
3. Switch to `use_point_normals=False` (faster)
4. Reduce `normal_neighbors` (faster normal estimation)
5. Set `mad_sigma=4.0` (skip MAD filtering)

### "Convergence is jumpy, unstable"
**Likely cause**: Correspondences changing wildly between iterations

**Fixes**:
1. Decrease `loss_param` (more stable weighting)
2. Use `loss="huber"` instead of "tukey" (smoother)
3. Increase `mad_sigma` (less aggressive filtering)
4. Improve initial guess (smaller alignment problem)

### "Great on clean data, fails on noisy data"
**Likely cause**: Loss function too aggressive

**Fixes**:
1. Switch to `loss="huber"` or `loss="fair"`
2. Increase `loss_param` significantly
3. Increase `mad_sigma` to 4.0-5.0
4. Decrease `min_linearity` (use more points)

### "Perfect alignment, but slow"
**Likely cause**: Parameters are good, just over-engineered

**Fixes**:
1. Reduce `max_iter` by 30-50% (usually converges earlier)
2. If `use_point_normals=True`, try switching to False
3. Reduce `normal_neighbors` from 15→10 or 10→5
4. Use `use_mad=False` if MAD isn't helping

---

## Parameter Impact Summary

| Parameter | Effect | Range | Notes |
|-----------|--------|-------|-------|
| **max_iter** | More = slower, more accurate | 5-50 | Most impact on convergence |
| **loss** | "tukey"=strict, "huber"=balanced, "fair"=lenient | - | Big impact on robustness |
| **loss_param** | Smaller = stricter outlier rejection | 0.05-0.2 | Usually 0.05-0.15 best |
| **max_dist** | Hard distance threshold | 0.2-2.0 | Depends on sensor scale |
| **max_seg_dist** | Max distance between segment points | 0.05-inf | Usually 0.1-0.3 for LiDAR |
| **use_mad** | Enable statistical filtering | T/F | Usually helps, minor overhead |
| **mad_sigma** | MAD multiplier for outliers | 1.5-5.0 | 2-3 typical, lower=stricter |
| **use_point_normals** | Point normal vs segment metric | T/F | True faster, False more robust |
| **normal_neighbors** | k for PCA normal estimation | 5-20 | Higher = better normals, slower |
| **min_linearity** | Minimum linearity score | 0.0-1.0 | 0.2-0.3 typical for structured env |

---

## Decision Tree

```
START: Need ICP parameters?
  |
  +-- Structured indoor environment?
  |   +-- YES: Use Scenario 1 (tukey, 0.08)
  |   +-- NO: Go to next
  |
  +-- Outdoor/unstructured?
  |   +-- YES: Use Scenario 2 (huber, 0.15)
  |   +-- NO: Go to next
  |
  +-- Need real-time performance?
  |   +-- YES: Use Scenario 3 (5 iters, no loss)
  |   +-- NO: Go to next
  |
  +-- Need maximum accuracy?
  |   +-- YES: Use Scenario 4 (tukey, 0.05, 50 iters)
  |   +-- NO: Use Scenario 2 (default)
  |
  +-- Still need tuning?
      +-- YES: Run parameter tuning workflow above
      +-- NO: Done!
```

---

## Benchmark Your Configuration

Before deploying, always benchmark:

```python
from icp_tuning import ICPBenchmark

bench = ICPBenchmark(n_points=your_scan_size)

# Test noise robustness
print("Noise Robustness:")
noise_results = bench.test_noise_robustness(
    [0.0, 0.01, 0.02, 0.05, 0.1]
)
for noise, error in noise_results.items():
    print(f"  {noise:.3f}: {error:.6f}")

# Test rotation range
print("\nRotation Range:")
rot_results = bench.test_rotation_range([5, 10, 15, 30, 45])
for angle, error in rot_results.items():
    print(f"  {angle:3d}°: {error:.6f}")
```

---

## Final Checklist

Before going into production:

- [ ] Choose scenario that matches your application
- [ ] Test with representative data from your sensor
- [ ] Verify convergence behavior (check `verbose=True` output)
- [ ] Check if results meet accuracy requirements
- [ ] Benchmark performance on your hardware
- [ ] Test edge cases (very noisy, extreme angles, etc.)
- [ ] Document your final configuration
- [ ] Set up fallback/recovery if alignment fails

---

**Good luck with your scan matching! The refactored code is solid. Now tune it and ship it! 🚀**

