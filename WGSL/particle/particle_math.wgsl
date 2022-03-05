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
}