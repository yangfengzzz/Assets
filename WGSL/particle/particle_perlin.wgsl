// Fast computation of x modulo 289
fn mod289Vec3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289Vec4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

// Compute indices for the PRNG
fn permute(x: vec4<f32>, uPerlinNoisePermutationSeed: f32) -> vec4<f32> {
    return mod289Vec4(((x*34.0)+1.0)*x + vec4<f32>(uPerlinNoisePermutationSeed));
}

// Quintic interpolant
fn fadeVec2(u: vec2<f32>) -> vec2<f32> {
    return u*u*u*(u*(u*6.0 - 15.0) + 10.0);
    
    // Original cubic interpolant (faster, but not 2nd order derivable)
    //return u*u*(3.0f - 2.0f*u);
}

fn fadeVec3(u: vec3<f32>) -> vec3<f32> {
    return u*u*u*(u*(u*6.0 - 15.0) + 10.0);
}

fn normalizeNoise(n: f32) -> f32 {
    // return noise in [0, 1]
    return 0.5*(2.44*n + 1.0);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
fn pnoise_gradients(pt: vec2<f32>, uPerlinNoisePermutationSeed: f32, gradients: ptr<function, vec4<f32> >, fpt: ptr<function, vec4<f32> >) {
    // Retrieve the integral part (for indexation)
    var ipt = floor(pt.xyxy) + vec4<f32>(0.0, 0.0, 1.0, 1.0);
    
    ipt = mod289Vec4(ipt);
    
    // Compute the 4 corners hashed gradient indices
    let ix = ipt.xzxz;
    let iy = ipt.yyww;
    let p = permute(permute(ix, uPerlinNoisePermutationSeed) + iy, uPerlinNoisePermutationSeed);
    
    // Fast version for :
    // p.x = P(P(ipt.x)      + ipt.y);
    // p.y = P(P(ipt.x+1.0f) + ipt.y);
    // p.z = P(P(ipt.x)      + ipt.y+1.0f);
    // p.w = P(P(ipt.x+1.0f) + ipt.y+1.0f);
    
    // With 'p', computes Pseudo Random Numbers
    let one_over_41 = 1.0 / 41.0; //0.02439f
    var gx = 2.0 * fract(p * one_over_41) - 1.0;
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    
    // Create unnormalized gradients
    var g00 = vec2<f32>(gx.x,gy.x);
    var g10 = vec2<f32>(gx.y,gy.y);
    var g01 = vec2<f32>(gx.z,gy.z);
    var g11 = vec2<f32>(gx.w,gy.w);
    
    // 'Fast' normalization
    let dp = vec4<f32>(dot(g00, g00), dot(g10, g10), dot(g01, g01), dot(g11, g11));
    let norm = inverseSqrt(dp);
    g00 = g00 * norm.x;
    g10 = g10 * norm.y;
    g01 = g01 * norm.z;
    g11 = g11 * norm.w;
    
    // Retrieve the fractional part (for interpolation)
    *fpt = fract(pt.xyxy) - vec4<f32>(0.0, 0.0, 1.0, 1.0);
    
    // Calculate gradient's influence
    let fx = (*fpt).xzxz;
    let fy = (*fpt).yyww;
    let n00 = dot(g00, vec2<f32>(fx.x, fy.x));
    let n10 = dot(g10, vec2<f32>(fx.y, fy.y));
    let n01 = dot(g01, vec2<f32>(fx.z, fy.z));
    let n11 = dot(g11, vec2<f32>(fx.w, fy.w));

    // Fast version for :
    // n00 = dot(g00, fpt + vec2(0.0f, 0.0f));
    // n10 = dot(g10, fpt + vec2(-1.0f, 0.0f));
    // n01 = dot(g01, fpt + vec2(0.0f,-1.0f));
    // n11 = dot(g11, fpt + vec2(-1.0f,-1.0f));
    
    *gradients = vec4<f32>(n00, n10, n01, n11);
}

// Classical Perlin Noise 2D
fn pnoise2D(pt: vec2<f32>, uPerlinNoisePermutationSeed: f32) -> f32 {
    var g:vec4<f32>;
    var fpt:vec4<f32>;
    pnoise_gradients(pt, uPerlinNoisePermutationSeed, &g, &fpt);
    
    // Interpolate gradients
    let u = fadeVec2(fpt.xy);
    let n1 = mix(g.x, g.y, u.x);
    let n2 = mix(g.z, g.w, u.x);
    let noise = mix(n1, n2, u.y);
    
    return noise;
}

// Derivative Perlin Noise 2D
fn dpnoise(pt: vec2<f32>, uPerlinNoisePermutationSeed: f32) -> vec3<f32> {
    var g:vec4<f32>;
    var fpt:vec4<f32>;
    pnoise_gradients(pt, uPerlinNoisePermutationSeed, &g, &fpt);
    
    let k0 = g.x;
    let k1 = g.y - g.x;
    let k2 = g.z - g.x;
    let k3 = g.x - g.z - g.y + g.w;
    var res = vec3<f32>(0.0);
    
    let u = fadeVec2(fpt.xy);
    res.x = k0 + k1*u.x + k2*u.y + k3*u.x*u.y;
    
    let dpt = 30.0*fpt.xy*fpt.xy*(fpt.xy*(fpt.xy - 2.0) + 1.0);
    res.y = dpt.x * (k1 + k3*u.y);
    res.z = dpt.y * (k2 + k3*u.x);
    
    return res;
}

// Classical Perlin Noise fbm 2D
fn fbm_pnoise2D(pt: vec2<f32>, zoom: f32, numOctave: u32, frequency: f32, amplitude: f32, uPerlinNoisePermutationSeed: f32) -> f32 {
    var sum = 0.0;
    var f = frequency;
    var w = amplitude;
    
    let v = zoom * pt;
    
    for (var i = 0u; i < numOctave; i = i + 1u) {
        sum = sum + w * pnoise2D(f*v, uPerlinNoisePermutationSeed);
        f = f * frequency;
        w = f * amplitude;
    }
    
    return sum;
}

// Derivative Perlin Noise fbm 2D
fn fbm_pnoise_derivative(pt: vec2<f32>, zoom: f32, numOctave: u32, frequency: f32, amplitude: f32, uPerlinNoisePermutationSeed: f32) -> f32 {
    var sum = 0.0;
    var f = frequency;
    var w = amplitude;

    var dn = vec2<f32>(0.0);
    
    let v = zoom * pt;
    
    for (var i = 0u; i < numOctave; i = i + 1u) {
        let n = dpnoise(f*v, uPerlinNoisePermutationSeed);
        dn = dn + n.yz;
        
        let crestFactor = 1.0 / (1.0 + dot(dn,dn));
        
        sum = sum + w * n.x * crestFactor;
        f = f * frequency;
        w = w * amplitude;
    }
    
    return sum;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Classical Perlin Noise 3D
fn pnoise3D(pt: vec3<f32>, uPerlinNoisePermutationSeed: f32) -> f32 {
    // Retrieve the integral part (for indexation)
    var ipt0 = floor(pt);
    var ipt1 = ipt0 + vec3<f32>(1.0);
    
    ipt0 = mod289Vec3(ipt0);
    ipt1 = mod289Vec3(ipt1);
    
    // Compute the 8 corners hashed gradient indices
    let ix = vec4<f32>(ipt0.x, ipt1.x, ipt0.x, ipt1.x);
    let iy = vec4<f32>(ipt0.yy, ipt1.yy);
    let p = permute(permute(ix, uPerlinNoisePermutationSeed) + iy, uPerlinNoisePermutationSeed);
    let p0 = permute(p + ipt0.zzzz, uPerlinNoisePermutationSeed);
    let p1 = permute(p + ipt1.zzzz, uPerlinNoisePermutationSeed);
    
    // Compute Pseudo Random Numbers
    var gx0 = p0 * (1.0 / 7.0);
    var gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    let gz0 = vec4<f32>(0.5) - abs(gx0) - abs(gy0);
    let sz0 = step(gz0, vec4<f32>(0.0));
    gx0 = gx0 - sz0 * (step(vec4<f32>(0.0), gx0) - 0.5);
    gy0 = gy0 - sz0 * (step(vec4<f32>(0.0), gy0) - 0.5);
    
    var gx1 = p1 * (1.0 / 7.0);
    var gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = fract(gx1);
    let gz1 = vec4<f32>(0.5) - abs(gx1) - abs(gy1);
    let sz1 = step(gz1, vec4<f32>(0.0));
    gx1 = gx1 - sz1 * (step(vec4<f32>(0.0), gx1) - 0.5);
    gy1 = gy1 - sz1 * (step(vec4<f32>(0.0), gy1) - 0.5);
    
    
    // Create unnormalized gradients
    var g000 = vec3<f32>(gx0.x, gy0.x, gz0.x);
    var g100 = vec3<f32>(gx0.y, gy0.y, gz0.y);
    var g010 = vec3<f32>(gx0.z, gy0.z, gz0.z);
    var g110 = vec3<f32>(gx0.w, gy0.w, gz0.w);
    var g001 = vec3<f32>(gx1.x, gy1.x, gz1.x);
    var g101 = vec3<f32>(gx1.y, gy1.y, gz1.y);
    var g011 = vec3<f32>(gx1.z, gy1.z, gz1.z);
    var g111 = vec3<f32>(gx1.w, gy1.w, gz1.w);
    
    // 'Fast' normalization
    var dp = vec4<f32>(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110));
    var norm = inverseSqrt(dp);
    g000 = g000 * norm.x;
    g100 = g100 * norm.y;
    g010 = g010 * norm.z;
    g110 = g110 * norm.w;
    
    dp = vec4<f32>(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111));
    norm = inverseSqrt(dp);
    g001 = g001 * norm.x;
    g101 = g101 * norm.y;
    g011 = g011 * norm.z;
    g111 = g111 * norm.w;
    
    // Retrieve the fractional part (for interpolation)
    let fpt0 = fract(pt);
    let fpt1 = fpt0 - vec3<f32>(1.0);
    
    // Calculate gradient's influence
    let n000 = dot(g000, fpt0);
    let n100 = dot(g100, vec3<f32>(fpt1.x, fpt0.yz));
    let n010 = dot(g010, vec3<f32>(fpt0.x, fpt1.y, fpt0.z));
    let n110 = dot(g110, vec3<f32>(fpt1.xy, fpt0.z));
    let n001 = dot(g001, vec3<f32>(fpt0.xy, fpt1.z));
    let n101 = dot(g101, vec3<f32>(fpt1.x, fpt0.y, fpt1.z));
    let n011 = dot(g011, vec3<f32>(fpt0.x, fpt1.yz));
    let n111 = dot(g111, fpt1);
    
    // Interpolate gradients
    let u = fadeVec3(fpt0);
    let nxy0 = mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y);
    let nxy1 = mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y);
    let noise = mix(nxy0, nxy1, u.z);
    
    return noise;
}

// Classical Perlin Noise 2D + time
fn pnoise_loop(u: vec2<f32>, dt: f32, uPerlinNoisePermutationSeed: f32) -> f32 {
    let pt1 = vec3<f32>(u, dt);
    let pt2 = vec3<f32>(u, dt - 1.0);
    
    return mix(pnoise3D(pt1, uPerlinNoisePermutationSeed), pnoise3D(pt2, uPerlinNoisePermutationSeed), dt);
}

// Classical Perlin Noise fbm 3D
fn fbm_pnoise3D(pt: vec3<f32>, zoom: f32, numOctave: u32, frequency: f32, amplitude: f32, uPerlinNoisePermutationSeed: f32) -> f32 {
    var sum = 0.0;
    var f = frequency;
    var w = amplitude;
    
    let v = zoom * pt;
    
    for (var i = 0u; i < numOctave; i = i + 1u) {
        sum = sum + w * pnoise3D(f*v, uPerlinNoisePermutationSeed);
        
        f = f * frequency;
        w = w * amplitude;
    }
    
    return sum;
}

fn fbm3D(ws: vec3<f32>, uPerlinNoisePermutationSeed: f32) -> f32 {
    let N = 128.0;
    let zoom = 1.0 / N;
    let octave = 4u;
    let freq = 2.0;
    let w = 0.45;
    
    return N * fbm_pnoise3D(ws, zoom, octave, freq, w, uPerlinNoisePermutationSeed);
}