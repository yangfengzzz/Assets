fn smoothstep_2(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (10.0 + t *(-15.0 + 6.0 * t));
}

fn ramp(x: f32) -> f32 {
    return smoothstep_2(-1.0, 1.0, x) * 2.0 - 1.0;
}

fn noise3d(seed: vec3<f32>, uPerlinNoisePermutationSeed: f32) -> vec3<f32> {
    return vec3<f32>(pnoise3D(seed, uPerlinNoisePermutationSeed),
                  pnoise3D(seed + vec3<f32>(31.416, -47.853, 12.793), uPerlinNoisePermutationSeed),
                  pnoise3D(seed + vec3<f32>(-233.145, -113.408, -185.31), uPerlinNoisePermutationSeed));
}

fn match_boundary(inv_noise_scale: f32, d: f32, normal: vec3<f32>, psi: ptr<function, vec3<f32> >) {
    let alpha = ramp(abs(d) * inv_noise_scale);
    let dp = dot(*psi, normal);
    *psi = mix(dp * normal, *psi, alpha);
}

// [ User customized sampling function ]
fn sample_potential(p:vec3<f32>, uPerlinNoisePermutationSeed:f32)->vec3<f32> {
    let num_octaves = 4u;
    
    // Potential
    var psi = vec3<f32>(0.0);
    
    // Compute normal and retrieve distance from colliders.
    var normal = vec3<f32>(0.0);
    let distance = compute_gradient(p, &normal);
    

    // let PlumeCeiling = 0.0;
    // let PlumeBase = -3.0;
    // let PlumeHeight = 80.0;
    // let RingRadius = 10.25;
    // let RingSpeed = 0.3;
    // let RingsPerSecond = 0.125;
    // let RingMagnitude = 10.0;
    // let RingFalloff = 0.7;

    
    var height_factor = 1.0;//ramp((p.y - PlumeBase)/ PlumeHeight);
    
    // Add turbulence octaves that respects boundaries.
    var noise_gain = 1.0;
    for(var i = 0u; i < num_octaves; i = i + 1u) {
        // const float noise_scale = 0.42f * noise_gain;
        let inv_noise_scale = 1.0 / noise_gain;
        
        let s = p * inv_noise_scale;
        let n = noise3d(s, uPerlinNoisePermutationSeed);
        
        match_boundary(inv_noise_scale, distance, normal, &psi);
        psi = psi + height_factor * noise_gain * n;

        noise_gain = noise_gain * 0.5;
    }
    
    // [ add custom potentials ]
    // --------
    // vec3 rising_force = vec3(-p.z, 0.0f, p.x);
    // 
    // let ring_y = PlumeCeiling;
    // let d = ramp(abs(distance) / RingRadius);
    // 
    // while (ring_y > PlumeBase) {
    // float ry = p.y - ring_y;
    // float rr = sqrt(dot(p.xz, p.xz));
    // vec3 v = vec3(rr-RingRadius, rr+RingRadius, ry);
    // float rmag = RingMagnitude / (dot(v,v) + RingFalloff);
    // vec3 rpsi = rmag * rising_force;
    // psi += mix(dot(rpsi, normal)*normal, psi, d);
    // ring_y -= RingSpeed / RingsPerSecond;
    // }
    
    return psi;
}


fn compute_curl(p: vec3<f32>, uPerlinNoisePermutationSeed: f32) -> vec3<f32> {
    let eps:f32 = 1.0e-4;
    
    let dx = vec3<f32>(eps, 0.0, 0.0);
    let dy = dx.yxy;
    let dz = dx.yyx;
    
    let p00 = sample_potential(p + dx, uPerlinNoisePermutationSeed);
    let p01 = sample_potential(p - dx, uPerlinNoisePermutationSeed);
    let p10 = sample_potential(p + dy, uPerlinNoisePermutationSeed);
    let p11 = sample_potential(p - dy, uPerlinNoisePermutationSeed);
    let p20 = sample_potential(p + dz, uPerlinNoisePermutationSeed);
    let p21 = sample_potential(p - dz, uPerlinNoisePermutationSeed);
    
    var v = vec3<f32>(0.0);
    v.x = p11.z - p10.z - p21.y + p20.y;
    v.y = p21.x - p20.x - p01.z + p00.z;
    v.z = p01.y - p00.y - p11.x + p10.x;
    v = v / (2.0*eps);
    
    return v;
}