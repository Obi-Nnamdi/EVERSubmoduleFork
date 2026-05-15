import safe_math;
import tensor_util;
import sh;

[AutoPyBindCUDA]
[CUDAKernel]
void intersect_sphere(
    TensorView<float> ray_origins,
    TensorView<float> ray_directions,
    TensorView<float> sphere_center,
    float sphere_radius,
    TensorView<float> T_values)
{
    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint prim_ind = dispatchIdx.x;

    // Coerce variables to standard representations
    if (prim_ind >= ray_origins.size(0)) {
        return;
    }

    float3 ray_o = get_float3(ray_origins, prim_ind);
    float3 ray_d = get_float3(ray_directions, prim_ind);
    float3 sphere_c = get_float3(sphere_center, 0);

    // Sphere interesct test

    // Sphere in same direction of ray? (t_center > 0)
    float3 L = sphere_c - ray_o;
    float t_center = dot(ray_d, L);

    if (t_center < 0) {
        return;
    }

    // Distance from ray to sphere center
    float L_dist_sq = dot(L, L);
    float d_squared = L_dist_sq - (t_center * t_center);

    float rad_squared = sphere_radius * sphere_radius;

    if (d_squared > rad_squared) {
        // Ray won't intersect sphere (too far away)
        return;
    }

    // T values (two of them for two intersections)
    float t_hc = sqrt(rad_squared - d_squared);
    float t_zero = t_center - t_hc;
    float t_one = t_center + t_hc;

    // Choose our chosen t value as the one closest to t_min without going over
    float t_min = 0; // TODO: Could be a parameter
    bool t_zero_valid = t_min <= t_zero;
    bool t_one_valid = t_min <= t_one;

    float chosen_t = -1;
    if (t_zero_valid && t_one_valid) {
        // TODO: I think it's always t_zero in this case
        chosen_t = min(t_zero, t_one);
    }
    else if (t_zero_valid) {
        chosen_t = t_zero;
    }
    else if (t_one_valid) {
        chosen_t = t_one;
    }
    else {
        // No valid t values
        return;
    }

    // Set values properly:
    T_values[prim_ind, 0] = chosen_t;
}