import safe_math;
import tensor_util;

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
    uint batch_ind = dispatchIdx.x;
    uint point_ind = dispatchIdx.y;
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