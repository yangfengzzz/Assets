// Map a range from [edge0, edge1] to [0, 1].
fn maprange(edge0: f32, edge1: f32, x: f32) -> f32 {
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

// Map a value in [0, 1] to peak at edge.
fn curve_inout(x: f32, edge: f32) -> f32 {
    // Coefficient for sub range.
    let a = maprange(0.0, edge, x);
    let b = maprange(edge, 1.0, x);
    
    // Quadratic ease-in / quadratic ease-out.
    let easein = a * (2.0 - a);        // a * a;
    let easeout = b*b - 2.0 * b + 1.0;  // 1.0f - b * b;
    
    // chose between easin / easout function.
    let result = mix(easein, easeout, step(edge, x));
    
    // Makes particles fade-in and out of existence
    return result;
}

fn compute_size(z: f32, decay: f32, uMinParticleSize: f32, uMaxParticleSize: f32) -> f32 {
    let min_size = uMinParticleSize;
    let max_size = uMaxParticleSize;
    
    // tricks to 'zoom-in' the pointsprite, just set to 1 to have normal size.
    let depth = (max_size-min_size) / (z);
    
    return mix(min_size, max_size, decay * depth);
}


fn base_color(position: vec3<f32>, decay: f32, uColorMode: u32, uBirthGradient: vec3<f32>, uDeathGradient: vec3<f32>) -> vec3<f32> {
    // Gradient mode
    if (uColorMode == 1u) {
        return mix(uBirthGradient, uDeathGradient, decay);
    }
    // Default mode
    return 0.5 * (normalize(position) + 1.0);
}

fn compute_color(base_color: vec3<f32>, decay: f32, texcoord: vec2<f32>, uFadeCoefficient: f32, uDebugDraw: bool) -> vec4<f32> {
    if (uDebugDraw) {
        return vec4<f32>(1.0);
    }
    
    var color = vec4<f32>(base_color, 1.0);
    
    // Centered coordinates.
    let p = 2.0 * (texcoord - 0.5);
    // Pixel intensity depends on its distance from center.
    let d = 1.0 - abs(dot(p, p));
    
    // Alpha coefficient.
    let alpha = smoothStep(0.0, 1.0, d);
    
    //color = texture(uSpriteSampler2d, texcoord).rrrr;
    color = color * alpha * decay * uFadeCoefficient;
    
    return color;
}