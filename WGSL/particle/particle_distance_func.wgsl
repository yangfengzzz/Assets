fn sample_distance(p: vec3<f32>) -> f32 {
    return p.y; //sdSphere(p - vec3(30.0f, 110.0f, 0.0f), 64.0);
}

fn compute_gradient(p: vec3<f32>, normal: ptr<private, vec3<f32> >) -> f32 {
    let d = sample_distance(p);
    
    let eps = vec2<f32>(0.01, 0.0);
    (*normal).x = sample_distance(p + eps.xyy) - d;
    (*normal).y = sample_distance(p + eps.yxy) - d;
    (*normal).z = sample_distance(p + eps.yyx) - d;
    (*normal) = normalize(*normal);
    
    return d;
}