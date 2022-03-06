fn pushParticle(position: vec3<f32>, velocity: vec3<f32>, age: f32, particles: ptr<storage, array<TParticle, 10> >, write_count: ptr<storage, atomic<u32> >) {
    // Emit particle id.
    let id = atomicAdd(write_count, 1u);
    
    var p = TParticle();
    p.position = vec4<f32>(position, 1.0);
    p.velocity = vec4<f32>(velocity, 0.0);
    p.start_age = age;
    p.age = age;
    p.id = id;
    
    (*particles)[id] = p;
}

fn createParticle(uEmitCount: u32, uEmitterType: u32, uEmitterPosition: vec3<f32>, uEmitterDirection: vec3<f32>, uEmitterRadius: f32, uParticleMinAge: f32, uParticleMaxAge: f32, write_count: ptr<storage, atomic<u32> >, particles: ptr<storage, array<TParticle, 10> >, randbuffer: array<f32, 10>, gid: u32) {
    // Random vector.
    let rid = 3u * gid;
    let rn = vec3<f32>(randbuffer[rid], randbuffer[rid+1u], randbuffer[rid+2u]);
    
    // Position
    var pos = uEmitterPosition;
    if (uEmitterType == 1u) {
        //pos += disk_distribution(uEmitterRadius, rn.xy);
        pos = pos + disk_even_distribution(uEmitterRadius, gid, uEmitCount);
    } else if (uEmitterType == 2) {
        pos = pos + sphere_distribution(uEmitterRadius, rn.xy);
    } else if (uEmitterType == 3) {
        pos = pos + ball_distribution(uEmitterRadius, rn);
    }
    
    // Velocity
    var vel = uEmitterDirection;
    
    // Age
    // The age is set by thread groups to assure we have a number of particles
    // factors of groupWidth, this method is safe but prevents continuous emission.
    // const float group_rand = randbuffer[gid];
    // [As the threadgroup are not full, some dead particles might appears if not
    // skipped in following stages].
    let single_rand = randbuffer[gid];
    
    let age = mix( uParticleMinAge, uParticleMaxAge, single_rand);
    
    pushParticle(pos, vel, age, particles, write_count);
}