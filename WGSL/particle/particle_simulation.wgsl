struct TParticle {
    position: vec4<f32>;
    velocity: vec4<f32>;
    start_age: f32;
    age: f32;
    padding0: f32; 
    id: u32;
};struct ParticleSimulationData {
    timeStep:f32;
    boundingVolumeType:i32;
    bboxSize:f32;
    scatteringFactor:f32;
    vectorFieldFactor:f32;
    curlNoiseFactor:f32;
    curlNoiseScale:f32;
    velocityFactor:f32;
};
struct Counter {
counter: atomic<u32>;
};
@group(0) @binding(17)
 var<uniform> u_simulationData: ParticleSimulationData;
 @group(0) @binding(21)
 var<storage, read_write> u_readAtomicBuffer: Counter;
 @group(0) @binding(22)
 var<storage, read_write> u_writeAtomicBuffer: Counter;
 @group(0) @binding(23)
 var<storage, read_write> u_readConsumeBuffer: array<TParticle, 1024>;
 @group(0) @binding(24)
 var<storage, read_write> u_writeConsumeBuffer: array<TParticle, 1024>;
 @group(0) @binding(16)
 var<uniform> u_randomBuffer: array<vec4<f32>, 512>;
 fn pi() -> f32 {
    return 3.141564;
}

fn twoPi() -> f32 {
    return 6.283185;
}

fn goldenAngle() -> f32 {
    return 2.399963;
}

fn cbrt(x: f32) -> f32 {
    return pow(x, 0.33333);
}

fn rotationX(c: f32, s: f32) -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0),
                        vec3<f32>(0.0, c, s),
                        vec3<f32>(0.0, -s, c));
}

fn rotationY(c: f32, s: f32) -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(c, 0.0, -s),
                        vec3<f32>(0.0, 1.0, 0.0),
                        vec3<f32>(s, 0.0, c));
}

fn rotationZ(c: f32, s: f32) -> mat3x3<f32> {
    return mat3x3<f32>(vec3<f32>(c, s, 0.0),
                        vec3<f32>(-s, c, 0.0),
                        vec3<f32>(0.0, 0.0, 1.0));
}

fn rotationXAngle(radians: f32) -> mat3x3<f32> {
    return rotationX(cos(radians), sin(radians));
}

fn rotationYAngle(radians: f32) -> mat3x3<f32> {
    return rotationY(cos(radians), sin(radians));
}

fn rotationZAngle(radians: f32) -> mat3x3<f32> {
    return rotationZ(cos(radians), sin(radians));
}

fn disk_distribution(radius: f32, rn: vec2<f32>) -> vec3<f32> {
    let r = radius * rn.x;
    let theta = twoPi() * rn.y;
    return vec3<f32>(r * cos(theta),
                    0.0,
                    r * sin(theta));
}

fn disk_even_distribution(radius: f32, id: u32, total: u32) -> vec3<f32> {
    // ref : http://blog.marmakoide.org/?p=1
    let theta:f32 = f32(id) * goldenAngle();
    let r = radius * sqrt(f32(id) / f32(total));
    return vec3<f32>(r * cos(theta),
                  0.0,
                  r * sin(theta));
}

fn sphere_distribution(radius: f32, rn: vec2<f32>) -> vec3<f32> {
    // ref : https://www.cs.cmu.edu/~mws/rpos.html
    //       https://gist.github.com/dinob0t/9597525
    let phi = twoPi() * rn.x;
    let z = radius * (2.0 * rn.y - 1.0);
    let r = sqrt(radius * radius - z * z);
    return vec3<f32>(r * cos(phi),
                    r * sin(phi),
                    z);
}

fn ball_distribution(radius: f32, rn: vec3<f32>) -> vec3<f32> {
    // ref : so@5408276
    let costheta = 2.0 * rn.x - 1.0;
    let phi = twoPi() * rn.y;
    let theta = acos(costheta);
    let r = radius * cbrt(rn.z);
    let s = sin(theta);
    
    return r * vec3<f32>(s * cos(phi),
                        s * sin(phi),
                        costheta);
}// Fast computation of x modulo 289
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
}fn smoothstep_2(edge0: f32, edge1: f32, x: f32) -> f32 {
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
        psi += height_factor * noise_gain * n;

        noise_gain = noise_gain * 0.5
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
    v /= (2.0*eps);
    
    return v;
}fn opUnion(d1: f32, d2: f32) -> f32 {
    return min(d1, d2);
}

fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
    let r = exp(-k*d1) + exp(-k*d2);
    return -log(r) / k;
}

fn opIntersection(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

fn opSubstraction(d1: f32, d2: f32) -> f32 {
    return max(d1, -d2);
}

fn opRepeat(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return p % c - 0.5*c;
}

fn opDisplacement(p: vec3<f32>, d: f32) -> f32 {
    var dp = d * p;
    return sin(dp.x)*sin(dp.y)*sin(dp.z);
}

fn sdPlane(p: vec3<f32>, n: vec4<f32>) -> f32 {
    //n.xyz = normalize(n.xyz);
    return n.w + dot(p, n.xyz);
}

fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn udRoundBox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    return length(max(abs(p)-b, vec3<f32>(0.0))) - r;
}

// fn sdCylinder(p: vec3<f32>, c: f32) -> f32 {
//     return length(p.xy) - c;
// }

fn sdCylinder(p: vec3<f32>, c: vec3<f32>) -> f32 {
    return opIntersection(length(p.xz-c.xy) - c.z, abs(p.y)-c.y);
}

fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
    let q = vec2<f32>(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}fn sample_distance(p: vec3<f32>) -> f32 {
    return p.y; //sdSphere(p - vec3(30.0f, 110.0f, 0.0f), 64.0);
}

fn compute_gradient(p: vec3<f32>, normal: ptr<function, vec3<f32> >) -> f32 {
    let d = sample_distance(p);
    
    let eps = vec2<f32>(0.01, 0.0);
    (*normal).x = sample_distance(p + eps.xyy) - d;
    (*normal).y = sample_distance(p + eps.yxy) - d;
    (*normal).z = sample_distance(p + eps.yyx) - d;
    (*normal) = normalize(*normal);
    
    return d;
}fn popParticle(index: u32) -> TParticle {{
    atomicSub(&u_readAtomicBuffer.counter, 1u);
    return u_readConsumeBuffer[index];
}}
fn pushParticle(p: TParticle) {
    let index = atomicAdd(&u_writeAtomicBuffer.counter, 1u);
    u_writeConsumeBuffer[index] = p;
}
fn updatedAge(p: TParticle, uTimeStep: f32) -> f32 {
    return clamp(p.age - uTimeStep, 0.0, p.start_age);
}
fn updateParticle(p: ptr<function, TParticle>, pos: vec3<f32>, vel: vec3<f32>, age: f32) {
    (*p).position.xyz = pos;
    (*p).velocity.xyz = vel;
    (*p).age = age;
}
fn calculateScattering(gid:u32)->vec3<f32> {
    var randforce = vec3<f32>(u_randomBuffer[gid], u_randomBuffer[gid+1u], u_randomBuffer[gid+2u]);
    randforce = 2.0 * randforce - 1.0;
    return u_simulationData.scatteringFactor * randforce;
}
fn calculateRepulsion(p:TParticle)->vec3<f32> {
    let push = vec3<f32>(0.0);    
    return push;
}
fn calculateTargetMesh(p:TParticle)->vec3<f32> {
    let pull = vec3<f32>(0.0);
    return pull;
}
fn calculateVectorField(p:TParticle,
                             uVectorFieldFactor:f32,
                            uVectorFieldTexture:texture_3d<f32>,
                             uVectorFieldSampler:sampler) {
    let dim = textureDimensions(uVectorFieldTexture);    let extent = vec3<f32>(0.5 * f32(dim.x), 0.5 * f32(dim.y), 0.5 * f32(dim.z));
    let texcoord = (p.position.xyz + extent) / (2.0 * extent);
    let vfield = textureSample(uVectorFieldTexture, uVectorFieldSampler, texcoord).xyz;
    
    return uVectorFieldFactor * vfield;
}
fn calculateCurlNoise(p:TParticle)->vec3<f32> {
    let curl_velocity = compute_curl(p.position.xyz * u_simulationData.curlNoiseScale, 0.0);
    return u_simulationData.curlNoiseFactor * curl_velocity;
}
fn collideSphere(r:f32, center:vec3<f32>, pos: ptr<function, vec3<f32> >, vel: ptr<function, vec3<f32> >) {
    let p = *pos - center;
    
    let dp = dot(p, p);
    let r2 = r*r;
    
    if (dp > r2) {
        let n = -p * inverseSqrt(dp);
        *vel = reflect(*vel, n);
        
        *pos = center - r*n;
    }
}
fn collideBox(corner:vec3<f32>, center:vec3<f32>, pos: ptr<function, vec3<f32> >, vel: ptr<function, vec3<f32> >) {
    let p = *pos - center;
    
    if (p.x < -corner.x) {
        p.x = -corner.x;
        *vel = reflect(*vel, vec3<f32>(1.0, 0.0, 0.0));
    }
    
    if (p.x > corner.x) {
        p.x = corner.x;
        *vel = reflect(*vel, vec3<f32>(-1.0, 0.0, 0.0));
    }
    
    if (p.y < -corner.y) {
        p.y = -corner.y;
        *vel = reflect(*vel, vec3<f32>(0.0, 1.0, 0.0));
    }
    
    if (p.y > corner.y) {
        p.y = corner.y;
        *vel = reflect(*vel, vec3<f32>(0.0, -1.0, 0.0));
    }
    
    if (p.z < -corner.z) {
        p.z = -corner.z;
        *vel = reflect(*vel, vec3<f32>(0.0, 0.0, 1.0));
    }
    
    if (p.z > corner.z) {
        p.z = corner.z;
        *vel = reflect(*vel, vec3<f32>(0.0, 0.0, -1.0));
    }
    
    *pos = p + center;
}
fn collisionHandling(pos: ptr<function, vec3<f32> >, vel: ptr<function, vec3<f32> >) {
    let r = 0.5 * u_simulationData.bboxSize;
    
    if (u_simulationData.boundingVolumeType == 0) {
        collideSphere(r, vec3<f32>(0.0), pos, vel);
    } else {
        if (u_simulationData.boundingVolumeType == 1) {
            collideBox(vec3<f32>(r), vec3<f32>(0.0), pos, vel); {
        }
    }
}
@stage(compute) @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let p = popParticle(gid);
    
    let age = updatedAge(p, u_simulationData.timeStep);
    
    if (age > 0.0) {
        // Calculate external forces.
        var force = vec3<f32>(0.0);
        
        force += calculateScattering(gid);
        force += calculateRepulsion(p);
        force += calculateTargetMesh(p);
        force += calculateCurlNoise(p);
        let dt = vec3<f32>(u_simulationData.timeStep);
        var velocity = p.velocity.xyz;
        var position = p.position.xyz;
        
        // Integrate velocity.
        velocity = fma(force, dt, velocity);
        velocity = u_simulationData.velocityFactor * normalize(velocity);
        position = fma(velocity, dt, position);
        
        // Handle collisions.
        collisionHandling(&position, &velocity);
        
        // Update the particle.
        updateParticle(p, position, velocity, age);
        
        // Save it in buffer.
        pushParticle(p);
    }
}