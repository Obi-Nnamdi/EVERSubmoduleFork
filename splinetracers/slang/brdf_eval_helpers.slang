import safe_math;
import tensor_util;

// Pattern from https://shader-slang.org/slang/docs/user-guide/a1-02-slangpy, section "Specializing shaders using slangtorch"
static const uint kMaxIncomingLightDirections = MAX_INCOMING_LIGHT_DIRECTIONS_FOR_LOOP_EVAL; // sent from slangtorch's 'defines' call.

// Blinn-Phong BRDF Eval using a single colored directional light and observer direction.
[Differentiable]
[BackwardDifferentiable]
float3 eval_blinn_phong_brdf_single_lighting_direction(
    float3 Kd, float3 Ks, float3 normal, float spec_c,
    no_diff float3 light_color, no_diff float3 light_dir, no_diff float3 outgoing_light_dir,
    out bool graceful_exit)
{
    graceful_exit = false;
    // Compute diffuse component
    float diffuse_term = dot(light_dir, normal);
    
    // If we're not on the same side of the normal, early exit
    if (diffuse_term < 0) {
        return {0, 0, 0};
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

    // Normal Return.
    graceful_exit = true;
    return diffuse_component + specular_component;
}

// Two different versions of the evaluation function.
// One uses gradient checkpointing with MaxIters, another uses
// recomputation.
#if USE_CHECKPOINTING_FOR_OUTGOING_COLOR_DIFF_BACKWARD_PASS
#warning "Using a Gradient Checkpointing Version of the Incoming Light Probe Kernel. Requires that there be a strict bound to the number of incoming light directions evaluated at once."
[Differentiable]
// Evaluate the given Blinn-Phong BRDF with incoming light given by a light probe and its associated index.
float3 blinn_phong_eval_rendered_color_from_probe_incoming_light(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    no_diff int probe_index,
    no_diff float3 outgoing_light_dir,
    float3 Kd,
    float3 Ks,
    float3 normal,
    float spec_c,
) {
    // Loop over all our incoming light directions and colors
    float3 full_color = { 0, 0, 0 };
    int valid_light_sources = 0;

    // [ForceUnroll(kMaxIncomingLightDirections)]
    [MaxIters(kMaxIncomingLightDirections)]
    for (no_diff uint light_ind = 0; light_ind < probe_incoming_light.size(1); light_ind++) {
        float3 light_color = no_diff get_float3(probe_incoming_light, probe_index, light_ind);
        float3 light_dir = no_diff get_float3(probe_incoming_light_dirs, light_ind); // Light dir never changes based on the point

        // Evaluate BRDF and track if it's even evaluable for the given lighting direction
        bool graceful_exit;
        float3 rendered_color = eval_blinn_phong_brdf_single_lighting_direction(Kd, Ks, normal, spec_c, light_color, light_dir, outgoing_light_dir, graceful_exit);

        // Accumulate the result and increase the amount of valid light sources if we've encountered a proper BRDF.
        if (graceful_exit) {
            full_color += rendered_color;
            valid_light_sources++;
        }
    }

    // Average using the amount of light sources contributing to our full color unless there weren't any (no NaNs), full color is 0
    if (valid_light_sources > 0) {
        full_color = safe_div(full_color, valid_light_sources);
    }

    return full_color;
}

#else

[BackwardDerivative(blinn_phong_eval_rendered_color_from_probe_incoming_light_bwd)]
// Evaluate the given Blinn-Phong BRDF with incoming light given by a light probe and its associated index.
// Uses custom derivative that forces recomputation of incoming light parameters.
float3 blinn_phong_eval_rendered_color_from_probe_incoming_light(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    no_diff int probe_index,
    no_diff float3 outgoing_light_dir,
    float3 Kd,
    float3 Ks,
    float3 normal,
    float spec_c,
) {
    // Loop over all our incoming light directions and colors
    float3 full_color = { 0, 0, 0 };
    int valid_light_sources = 0;

    // Loop doesn't need MaxIterations specifier
    for (no_diff uint light_ind = 0; light_ind < probe_incoming_light.size(1); light_ind++) {
        float3 light_color = no_diff get_float3(probe_incoming_light, probe_index, light_ind);
        float3 light_dir = no_diff get_float3(probe_incoming_light_dirs, light_ind); // Light dir never changes based on the point

        // Evaluate BRDF and track if it's even evaluable for the given lighting direction
        bool graceful_exit;
        float3 rendered_color = eval_blinn_phong_brdf_single_lighting_direction(Kd, Ks, normal, spec_c, light_color, light_dir, outgoing_light_dir, graceful_exit);

        // Accumulate the result and increase the amount of valid light sources if we've encountered a proper BRDF.
        if (graceful_exit) {
            full_color += rendered_color;
            valid_light_sources++;
        }
    }

    // Average using the amount of light sources contributing to our full color unless there weren't any (no NaNs), full color is 0
    if (valid_light_sources > 0) {
        full_color = safe_div(full_color, valid_light_sources);
    }

    return full_color;
}

void blinn_phong_eval_rendered_color_from_probe_incoming_light_bwd(
    TensorView<float> probe_incoming_light,
    TensorView<float> probe_incoming_light_dirs,
    no_diff int probe_index,
    no_diff float3 outgoing_light_dir,
    inout DifferentialPair<float3> Kd,
    inout DifferentialPair<float3> Ks,
    inout DifferentialPair<float3> normal,
    inout DifferentialPair<float> spec_c,
    float3 dLdOutput
) {
    // Partially Recompute forward pass (no checkpointing)
    int valid_light_sources = 0;

    for (uint light_ind = 0; light_ind < probe_incoming_light.size(1); light_ind++) {
        float3 light_color = get_float3(probe_incoming_light, probe_index, light_ind);
        float3 light_dir = get_float3(probe_incoming_light_dirs, light_ind); // Light dir never changes based on the point

        // Evaluate BRDF and track if it's even evaluable for the given lighting direction
        bool valid_brdf = dot(light_dir, normal.p) >= 0;
        // Alternative idea if valid_brdf evaluation gets more complex
        // float3 rendered_color = eval_blinn_phong_brdf_single_lighting_direction(Kd.p, Ks.p, normal.p, spec_c.p,
        // light_color, light_dir, outgoing_light_dir, valid_brdf);

        // Accumulate the result and increase the amount of valid light sources if we've encountered a proper BRDF.
        if (valid_brdf) {
            valid_light_sources++;

            // Gradients updated this way to prevent overwriting.
            DifferentialPair<float3> temp_Kd = diffPair(Kd.p); // automatically zero
            DifferentialPair<float3> temp_Ks = diffPair(Ks.p);
            DifferentialPair<float3> temp_normal = diffPair(normal.p);
            DifferentialPair<float> temp_spec_c = diffPair(spec_c.p);
            float3 dSumdBRDF = { 1.0, 1.0, 1.0 }; // Always 1 (no weighted average/sum)
            
            // Calculate the gradients dBRDF(light_i) / dComponent
            bwd_diff(eval_blinn_phong_brdf_single_lighting_direction)(temp_Kd, temp_Ks, temp_normal, temp_spec_c,
                light_color, light_dir, outgoing_light_dir, dSumdBRDF * dLdOutput);

            // Update gradients
            Kd = diffPair(Kd.p, Kd.d + temp_Kd.d);
            Ks = diffPair(Ks.p, Ks.d + temp_Ks.d);
            normal = diffPair(normal.p, normal.d + temp_normal.d);
            spec_c = diffPair(spec_c.p, spec_c.d + temp_spec_c.d);
        }
    }

    // If no valid output was generated, there is no gradient
    if (valid_light_sources == 0) {
        Kd = diffPair(Kd.p, { 0, 0, 0 });
        Ks = diffPair(Ks.p, { 0, 0, 0 });
        normal = diffPair(normal.p, { 0, 0, 0 });
        spec_c = diffPair(spec_c.p, 0.);
        
        return;
    }

    // Determine final gradients and return them:
    // Our Kd -> spec_c parameters now store accumulated gradients for
    // dSum/dKd -> dSum/dspec_c by adding over all the valid light sources
    // Now that we know how many valid light sources we had, we can apply the
    // preceding dAverage/dSum and dL/dAverage chain rule gradient:
    // e.g: (Note that Average == Output from forward pass)
    // dL / dKd = (dL/dAverage) * (dAverage/dSum) * (dSum/dKd)
    //              { (yes) }       { (no) }        { (yes) }
    // 
    // where dAverage / dSum = 1 / valid_light_sources

    Kd = diffPair(Kd.p, safe_div(Kd.d, valid_light_sources));
    Ks = diffPair(Ks.p, safe_div(Ks.d, valid_light_sources));
    normal = diffPair(normal.p, safe_div(normal.d, valid_light_sources));
    spec_c = diffPair(spec_c.p, safe_div(spec_c.d, valid_light_sources));
}

#endif