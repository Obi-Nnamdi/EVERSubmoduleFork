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
        incoming_light_dirs: (P, N, 3) Directions oriented towards the light source in world space
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
    float3 light_dir = no_diff get_float3(incoming_light_dirs, point_ind, light_ind); // TODO: Could be reduced from (P, N, 3) -> (N, 3) to save memory

    
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
    float specular_term = pow(saturate(half_dot_normal), spec_c); // clip to within 0 -> 1 range to prevent NaNs.

    float3 specular_intensity = light_color * Ks;
    float3 specular_component = specular_term * specular_intensity;

    // Combine components and store the result
    float3 full_color = diffuse_component + specular_component;
    output.storeOnce(uint3(point_ind, light_ind, 0), full_color.r);
    output.storeOnce(uint3(point_ind, light_ind, 1), full_color.g);
    output.storeOnce(uint3(point_ind, light_ind, 2), full_color.b);
}