struct TParticle {
    position: vec4<f32>;
    velocity: vec4<f32>;
    start_age: f32;
    age: f32;
    padding0: f32; 
    id: u32;
};struct ParticleEmitterData {
    emitterPosition: vec3<f32>;
    emitCount: u32;
    emitterDirection: vec3<f32>;
    emitterType: u32;
    emitterRadius: f32;
    particleMinAge: f32;
    particleMaxAge: f32;
    pad: f32;
};
struct Counter {
counter: atomic<u32>;
};
@group(0) @binding(18)
 var<uniform> u_emitterData: ParticleEmitterData;
 @group(0) @binding(21)
 var<storage, read_write> u_readAtomicBuffer: Counter;
 @group(0) @binding(23)
 var<storage, read_write> u_readConsumeBuffer: array<TParticle, 32768>;
 @group(0) @binding(16)
 var<uniform> u_randomBuffer: array<vec4<f32>, 256>;
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
}@stage(compute) @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
) {
if (global_id.x < u_emitterData.emitCount) {
 // Random vector.
    let rn = vec3<f32>(u_randomBuffer[global_id.x / 256u].x, u_randomBuffer[global_id.x / 256u].y, u_randomBuffer[global_id.x / 256u].z);
    
    // Position
    var pos = u_emitterData.emitterPosition;
    if (u_emitterData.emitterType == 1u) {
        //pos += disk_distribution(uEmitterRadius, rn.xy);
        pos = pos + disk_even_distribution(u_emitterData.emitterRadius, global_id.x, u_emitterData.emitCount);
    } else if (u_emitterData.emitterType == 2u) {
        pos = pos + sphere_distribution(u_emitterData.emitterRadius, rn.xy);
    } else if (u_emitterData.emitterType == 3u) {
        pos = pos + ball_distribution(u_emitterData.emitterRadius, rn);
    }
    
    // Velocity
    var vel = u_emitterData.emitterDirection;
    
    // Age
    // The age is set by thread groups to assure we have a number of particles
    // factors of groupWidth, this method is safe but prevents continuous emission.
    // const float group_rand = randbuffer[gid];
    // [As the threadgroup are not full, some dead particles might appears if not
    // skipped in following stages].
    
    let age = mix( u_emitterData.particleMinAge, u_emitterData.particleMaxAge, u_randomBuffer[global_id.x / 256u].w);

    // Emit particle id.
    let id = atomicAdd(&u_readAtomicBuffer.counter, 1u);
    
    var p = TParticle();
    p.position = vec4<f32>(pos, 1.0);
    p.velocity = vec4<f32>(vel, 0.0);
    p.start_age = age;
    p.age = age;
    p.id = id;
    
    u_readConsumeBuffer[id] = p;
}
}