@group(0) @binding(14)
 var<uniform> u_baseColor: vec4<f32>;
 @group(0) @binding(13)
 var<uniform> u_alphaCutoff: f32;
 @group(0) @binding(15) var u_baseTexture: texture_2d<f32>;
 @group(0) @binding(16) var u_baseSampler: sampler;
 struct Output {
@location(0) finalColor: vec4<f32>;
}
struct VertexOut {
@location(0) v_uv: vec2<f32>;
}
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
var baseColor = u_baseColor;
var textureColor = textureSample(u_baseTexture, u_baseSampler, in.v_uv);
baseColor = baseColor * textureColor;
out.finalColor = baseColor;
return out;
}