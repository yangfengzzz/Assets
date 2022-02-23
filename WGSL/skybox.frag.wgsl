let PI:f32 = 3.14159265359;
let RECIPROCAL_PI:f32 = 0.31830988618;
let EPSILON:f32 = 1.0e-6;
let LOG2:f32 = 1.442695;
@group(0) @binding(0) var u_cubeTexture: texture_cube<f32>;
 @group(0) @binding(1) var u_cubeSampler: sampler;
 struct Output {
@location(0) finalColor: vec4<f32>;
}
struct VertexOut {
@location(0) v_cubeUV: vec3<f32>;
}
fn saturate(a:f32)->f32 { return clamp( a, 0.0, 1.0 );}
fn whiteCompliment(a:f32)->f32 { return 1.0 - saturate( a );}
fn RGBMToLinear(value: vec4<f32>, maxRange: f32)-> vec4<f32> {
    return vec4<f32>( value.rgb * value.a * maxRange, 1.0 );
}
fn gammaToLinear(srgbIn: vec4<f32>)-> vec4<f32> {
    return vec4<f32>( pow(srgbIn.rgb, vec3<f32>(2.2)), srgbIn.a);
}
fn linearToGamma(linearIn: vec4<f32>)-> vec4<f32> {
    return vec4<f32>( pow(linearIn.rgb, vec3<f32>(1.0 / 2.2)), linearIn.a);
}
@stage(fragment)
fn main(in: VertexOut, ) -> Output {
var out:Output;
var textureColor = textureSample( u_cubeTexture, u_cubeSampler, in.v_cubeUV );
out.finalColor = textureColor;
return out;
}
