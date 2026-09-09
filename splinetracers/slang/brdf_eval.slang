import safe_math;
import tensor_util;
import brdf_eval_helpers;

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
/**
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    Good code reference: https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

    Inputs:
        incoming_light: (P, N, 3) R,G,B values of incoming light for each direction
        incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
        normal: (P, 3) Surface normals in world space for each of P points
        outgoing_directions: (P, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
        diffuse_K: (P, 3)
        specular_K: (P, 3)
        spec_reflect_c: (P,)

    Outputs:
        color: (P, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions
        (TODO, if necessary) -  positive_contribs: (P, N) bool values indicating whether the lighting contribution for that point and light direction
                           is on the positive side of the plane defined by the normal.

*/
void eval_outgoing_radiance_blinn_phong(
    TensorView<float> incoming_light,
    TensorView<float> incoming_light_dirs,
    TensorView<float> outgoing_directions,
    DiffTensorView<float> normals,
    DiffTensorView<float> diffuse_K,
    DiffTensorView<float> specular_K,
    DiffTensorView<float> spec_reflect_c,
    DiffTensorView<float> output
)
{
    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint point_ind = dispatchIdx.x;
    uint light_ind = dispatchIdx.y;

    // Early exit if there's no data to process
    if (point_ind >= incoming_light.size(0)) {
        return;
    }
    if (light_ind >= incoming_light.size(1)) {
        return;
    }

    // Coerce variables to standard representations
    float3 Kd = get_float3(diffuse_K, point_ind);
    float3 Ks = get_float3(specular_K, point_ind);
    float3 normal = get_float3(normals, point_ind);
    float spec_c = spec_reflect_c[point_ind];

    float3 outgoing_light_dir = no_diff get_float3(outgoing_directions, point_ind);

    float3 light_color = no_diff get_float3(incoming_light, point_ind, light_ind);
    float3 light_dir = no_diff get_float3(incoming_light_dirs, light_ind); // Light dir never changes based on the point

    // Compute diffuse component
    float diffuse_term = dot(light_dir, normal);
    // If we're not on the same side of the normal, early exit
    if (diffuse_term < 0) {
        return;
    }

    float3 diffuse_intensity = light_color * Kd;

    // NOTE: Diffuse term could be clipped similar to specular, but it should always be less than 1 since light_dir and normal are both normalized before input.
    float3 diffuse_component = diffuse_term * diffuse_intensity;

    // Compute Specular component
    float3 halfway_vec = l2_normalize(light_dir + outgoing_light_dir);
    float half_dot_normal = dot(halfway_vec, normal);

    // TODO: A "gentler" clipping (sigmoid, etc.) might be preferred for gradient propagation
    float specular_term = safe_pow_pos_b(saturate(half_dot_normal), spec_c); // clip exponent to within 0 -> 1 range and don't do
                                                                             // POWs on really small numbers (assuming pos b exponent) to prevent NaNs.

    float3 specular_intensity = light_color * Ks;
    float3 specular_component = specular_term * specular_intensity;

    // Combine components and store the result
    float3 full_color = diffuse_component + specular_component;
    output.storeOnce(uint3(point_ind, light_ind, 0), full_color.r);
    output.storeOnce(uint3(point_ind, light_ind, 1), full_color.g);
    output.storeOnce(uint3(point_ind, light_ind, 2), full_color.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
/**
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    Good code reference: https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

    Inputs:
        probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
        probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
        incoming_light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
        outgoing_directions: (B, HW, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
        normals: (B, HW, 3) Surface normals in world space for each of HW points
        diffuse_K: (B, HW, 3)
        specular_K: (B, HW, 3)
        spec_reflect_c: (B, HW,)

    Outputs:
        color: (B, HW, N, 3) R,G,B values of lighting contributions at each of P points for all of the N directions for every camera

*/
void eval_outgoing_radiance_blinn_phong_with_incoming_light_cache(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    TensorView<int32_t> incoming_light_probe_query,
    TensorView<float> outgoing_directions,
    DiffTensorView<float> normals,
    DiffTensorView<float> diffuse_K,
    DiffTensorView<float> specular_K,
    DiffTensorView<float> spec_reflect_c,
    DiffTensorView<float> output
)
{
    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // Awkward looking that the Batch dim isn't the first dim, but it's to prevent limitations of block size (point dim is by far the largest.)
    uint batch_ind = dispatchIdx.y;
    uint point_ind = dispatchIdx.x;
    uint light_ind = dispatchIdx.z;

    // Early exit if there's no data to process
    if (batch_ind >= normals.size(0)) {
        return;
    }
    if (point_ind >= normals.size(1)) {
        return;
    }
    if (light_ind >= probe_incoming_light.size(1)) {
        return;
    }

    // Coerce variables to standard representations
    float3 Kd = get_float3(diffuse_K, batch_ind, point_ind);
    float3 Ks = get_float3(specular_K, batch_ind, point_ind);
    float3 normal = get_float3(normals, batch_ind, point_ind);
    float spec_c = spec_reflect_c[batch_ind, point_ind];

    float3 outgoing_light_dir = no_diff get_float3(outgoing_directions, batch_ind, point_ind);

    // Get the index with which we're indexing the "P" dimension of the probe with
    int probe_index = incoming_light_probe_query[batch_ind, point_ind];

    float3 light_color = no_diff get_float3(probe_incoming_light, probe_index, light_ind);
    float3 light_dir = no_diff get_float3(probe_incoming_light_dirs, light_ind); // Light dir never changes based on the point

    // Compute diffuse component
    float diffuse_term = dot(light_dir, normal);
    // If we're not on the same side of the normal, early exit
    if (diffuse_term < 0) {
        return;
    }

    float3 diffuse_intensity = light_color * Kd;

    // NOTE: Diffuse term could be clipped similar to specular, but it should always be less than 1 since light_dir and normal are both normalized before input.
    float3 diffuse_component = diffuse_term * diffuse_intensity;

    // Compute Specular component
    float3 halfway_vec = l2_normalize(light_dir + outgoing_light_dir);
    float half_dot_normal = dot(halfway_vec, normal);

    // TODO: A "gentler" clipping (sigmoid, etc.) might be preferred for gradient propagation
    float specular_term = safe_pow_pos_b(saturate(half_dot_normal), spec_c); // clip exponent to within 0 -> 1 range and don't do
                                                                             // POWs on really small numbers (assuming pos b exponent) to prevent NaNs.

    float3 specular_intensity = light_color * Ks;
    float3 specular_component = specular_term * specular_intensity;

    // Combine components and store the result
    float3 full_color = diffuse_component + specular_component;
    output.storeOnce(uint4(batch_ind, point_ind, light_ind, 0), full_color.r);
    output.storeOnce(uint4(batch_ind, point_ind, light_ind, 1), full_color.g);
    output.storeOnce(uint4(batch_ind, point_ind, light_ind, 2), full_color.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
/**
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding incoming light.
    Computes all intermediate light ray contributions within the kernel to save memory.
    Good code reference: https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

    Inputs:
        probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
        probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point)
        incoming_light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
        outgoing_directions: (B, HW, 3) Outgoing (view) direction of radiance, in world space (head at the camera, tip at the surface point) for each point.
        normals: (B, HW, 3) Surface normals in world space for each of HW points
        diffuse_K: (B, HW, 3)
        specular_K: (B, HW, 3)
        spec_reflect_c: (B, HW,)

    Outputs:
        color: (B, HW, 3) R,G,B values of comulative lighting contributions at each of the HW points and batch size.

*/
void eval_outgoing_radiance_blinn_phong_with_incoming_light_cache_mem_save(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    TensorView<int32_t> incoming_light_probe_query,
    TensorView<float> outgoing_directions,
    DiffTensorView<float> normals,
    DiffTensorView<float> diffuse_K,
    DiffTensorView<float> specular_K,
    DiffTensorView<float> spec_reflect_c,
    DiffTensorView<float> output
)
{
    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // Awkward looking that the Batch dim isn't the first dim, but it's to prevent limitations of block size (point dim is by far the largest.)
    uint batch_ind = dispatchIdx.y;
    uint point_ind = dispatchIdx.x;

    // Early exit if there's no data to process
    if (batch_ind >= normals.size(0)) {
        return;
    }
    if (point_ind >= normals.size(1)) {
        return;
    }

    // Coerce variables to standard representations
    float3 Kd = get_float3(diffuse_K, batch_ind, point_ind);
    float3 Ks = get_float3(specular_K, batch_ind, point_ind);
    float3 normal = get_float3(normals, batch_ind, point_ind);
    float spec_c = spec_reflect_c[batch_ind, point_ind];

    float3 outgoing_light_dir = no_diff get_float3(outgoing_directions, batch_ind, point_ind);

    // Get the index with which we're indexing the "P" dimension of the probe with
    int probe_index = incoming_light_probe_query[batch_ind, point_ind];

    float3 full_color = blinn_phong_eval_rendered_color_from_probe_incoming_light(
        probe_incoming_light, probe_incoming_light_dirs, probe_index,
        outgoing_light_dir, Kd, Ks, normal, spec_c
    );

    output.storeOnce(uint3(batch_ind, point_ind, 0), full_color.r);
    output.storeOnce(uint3(batch_ind, point_ind, 1), full_color.g);
    output.storeOnce(uint3(batch_ind, point_ind, 2), full_color.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
/**
    Evaluate the given Blinn-Phong BRDFs with the given incoming light, matching each point to its corresponding outgoing light.
    Computes all intermediate light ray contributions within the kernel to save memory.
    Good code reference: https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model

    Inputs:
        probe_incoming_light: (P, N, 3) R,G,B values of incoming light for each direction in the light probe
        probe_incoming_light_dirs: (N, 3) Directions oriented towards the light source in world space (assumed to be the same for each point).
        probe_outgoing_light: (P, O, 3) R,G,B values of outgoing light for each direction
        probe_outgoing_light_dirs: (O, 3) All extra outgoing (view) directions of radiance (essentially just reversed probe_incoming_light_dirs)
                                    Head at the camera, tip at the surface point.
        light_probe_query: (B, HW) Correspondence between each of the HW points and their probe number.
        normals: (B, HW, 3) Surface normals in world space for each of HW points
        diffuse_K: (B, HW, 3)
        specular_K: (B, HW, 3)
        spec_reflect_c: (B, HW,)

    Outputs:
        color_diff: (B, HW, 3) R,G,B values that specify the difference in color between the outgoing light probe and the rendered BRDF colors
        for each point (specified by Batch Size B and HW), averaged across all the probe outgoing light directions.
*/
void eval_outgoing_radiance_at_multiple_directions_with_probe(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    TensorView<float> probe_outgoing_light,
    TensorView<float> probe_outgoing_light_dirs,
    TensorView<int32_t> light_probe_query,
    DiffTensorView<float> normals,
    DiffTensorView<float> diffuse_K,
    DiffTensorView<float> specular_K,
    DiffTensorView<float> spec_reflect_c,
    DiffTensorView<float> output,
)
{
    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // Awkward looking that the Batch dim isn't the first dim, but it's to prevent limitations of block size (point dim is by far the largest.)
    uint batch_ind = dispatchIdx.y;
    uint point_ind = dispatchIdx.x;

    // Early exit if there's no data to process
    if (batch_ind >= normals.size(0)) {
        return;
    }
    if (point_ind >= normals.size(1)) {
        return;
    }

    // Coerce variables to standard representations
    float3 Kd = get_float3(diffuse_K, batch_ind, point_ind);
    float3 Ks = get_float3(specular_K, batch_ind, point_ind);
    float3 normal = get_float3(normals, batch_ind, point_ind);
    float spec_c = spec_reflect_c[batch_ind, point_ind];

    // Get the index with which we're indexing the "P" dimension of the probe with
    int probe_index = light_probe_query[batch_ind, point_ind];
    int num_outgoing_directions = probe_outgoing_light.size(1);

    // Iteratively compute average color diff between outgoing light directions and outgoing light
    float3 running_color_diff = { 0, 0, 0 };
    for (uint outgoing_light_ind = 0; outgoing_light_ind < num_outgoing_directions; outgoing_light_ind++) {
        float3 outgoing_light_dir = get_float3(probe_outgoing_light_dirs, outgoing_light_ind);
        float3 target_color = get_float3(probe_outgoing_light, probe_index, outgoing_light_ind);

        float3 rendered_color = blinn_phong_eval_rendered_color_from_probe_incoming_light(
            probe_incoming_light, probe_incoming_light_dirs, probe_index,
            outgoing_light_dir, Kd, Ks, normal, spec_c
        );

        float3 color_diff = target_color - rendered_color;
        running_color_diff += color_diff * color_diff; // add MSE
    }

    // Average across "O" dimension
    float3 average_color_diff = safe_div(running_color_diff, num_outgoing_directions);
    output.storeOnce(uint3(batch_ind, point_ind, 0), average_color_diff.r);
    output.storeOnce(uint3(batch_ind, point_ind, 1), average_color_diff.g);
    output.storeOnce(uint3(batch_ind, point_ind, 2), average_color_diff.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
void eval_outgoing_radiance_at_multiple_directions_with_probe_bwd(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    TensorView<float> probe_outgoing_light,
    TensorView<float> probe_outgoing_light_dirs,
    TensorView<int32_t> light_probe_query,
    TensorView<float> normals,
    TensorView<float> normals_grad,
    TensorView<float> diffuse_K,
    TensorView<float> diffuse_K_grad,
    TensorView<float> specular_K,
    TensorView<float> specular_K_grad,
    TensorView<float> spec_reflect_c,
    TensorView<float> spec_reflect_c_grad,
    TensorView<float> output,
    TensorView<float> output_grad,
) {
    // normals.diff
    // normals.primal
    // https://docs.shader-slang.org/en/stable/auto-diff-tutorial-1.html#backward-mode-differentiation
    // Goal: find [dBRDF/dnormal, dBRDF/dKd, dBRDF/dKs, dBRDF/dspec_c

    // Initial setup
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // Awkward looking that the Batch dim isn't the first dim, but it's to prevent limitations of block size (point dim is by far the largest.)
    uint batch_ind = dispatchIdx.y;
    uint point_ind = dispatchIdx.x;

    // Early exit if there's no data to process
    if (batch_ind >= normals.size(0)) {
        return;
    }
    if (point_ind >= normals.size(1)) {
        return;
    }

    // RECOMPUTE FORWARD PASS (then multiply by output_grad):
    // Coerce variables to standard representations
    float3 Kd = get_float3(diffuse_K, batch_ind, point_ind);
    float3 Ks = get_float3(specular_K, batch_ind, point_ind);
    float3 normal = get_float3(normals, batch_ind, point_ind);
    float spec_c = spec_reflect_c[batch_ind, point_ind];

    // Get the index with which we're indexing the "P" dimension of the probe with
    int probe_index = light_probe_query[batch_ind, point_ind];
    int num_outgoing_directions = probe_outgoing_light.size(1);

    // Get current output derivative (dL / dAvg)
    float3 dL_dAvg = get_float3(output_grad, batch_ind, point_ind);

    // Recompute forward pass to get values to work with
    for (uint outgoing_light_ind = 0; outgoing_light_ind < num_outgoing_directions; outgoing_light_ind++) {
        float3 outgoing_light_dir = get_float3(probe_outgoing_light_dirs, outgoing_light_ind);
        float3 target_color = get_float3(probe_outgoing_light, probe_index, outgoing_light_ind);

        float3 rendered_color = blinn_phong_eval_rendered_color_from_probe_incoming_light(
            probe_incoming_light, probe_incoming_light_dirs, probe_index,
            outgoing_light_dir, Kd, Ks, normal, spec_c
        );

        // Recompute color_diff
        float3 color_diff = target_color - rendered_color;

        // Compute dAvg / drendered_color_i
        // = (dAvg / dSum) * (dSum / d(diff^2)) * (d (diff^2) / (d_diff)) * (dDiff / dRendered)
        float3 dAvgdiff_dRendered = safe_div(-2 * color_diff, num_outgoing_directions);
        float3 dL_dRendered = dL_dAvg * dAvgdiff_dRendered; // chain rule

        // Compute our effect on each parameter of the rendered color
        var Kd_diff = diffPair(Kd);
        var Ks_diff = diffPair(Ks);
        var normal_diff = diffPair(normal);
        var spec_c_diff = diffPair(spec_c);
        bwd_diff(blinn_phong_eval_rendered_color_from_probe_incoming_light_fwd)(probe_incoming_light, probe_incoming_light_dirs, probe_index, outgoing_light_dir,
                                                                            Kd_diff, Ks_diff, normal_diff, spec_c_diff, dL_dRendered);

        // Add these to our derivatives (interlocked add not needed as everything's on one thread)
        add_float3(normals_grad, batch_ind, point_ind, normal_diff.d);
        add_float3(diffuse_K_grad, batch_ind, point_ind, Kd_diff.d);
        add_float3(specular_K_grad, batch_ind, point_ind, Ks_diff.d);
        spec_reflect_c_grad[uint2(batch_ind, point_ind)] += spec_c_diff.d;
        // spec_reflect_c_grad[uint2(batch_ind, point_ind)] += dL_dRendered.x;
        // var temp = 0.;
        // spec_reflect_c_grad.InterlockedAdd(uint2(batch_ind, point_ind), dL_dRendered.x, temp);
    }
}