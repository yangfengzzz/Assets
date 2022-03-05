fn opUnion(d1: f32, d2: f32) -> f32 {
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
}