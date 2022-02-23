struct CameraData {
 u_viewMat: mat4x4<f32>;
 u_projMat: mat4x4<f32>;
 u_VPMat: mat4x4<f32>;
 u_viewInvMat: mat4x4<f32>;
 u_projInvMat: mat4x4<f32>;
 u_cameraPos: vec3<f32>;
}
struct RendererData {
 u_localMat: mat4x4<f32>;
 u_modelMat: mat4x4<f32>;
 u_MVMat: mat4x4<f32>;
 u_MVPMat: mat4x4<f32>;
 u_MVInvMat: mat4x4<f32>;
 u_normalMat: mat4x4<f32>;
}
@group(0) @binding(11)
 var<uniform> u_cameraData: CameraData;
 @group(0) @binding(23)
 var<uniform> u_rendererData: RendererData;
 @group(0) @binding(13)
 var<uniform> u_tilingOffset: vec4<f32>;

 struct VertexOut {
@location(0) v_uv: vec2<f32>;
@location(1) v_normal: vec3<f32>;
@location(2) v_pos: vec3<f32>;
@builtin(position) position: vec4<f32>;
}

struct VertexIn {
@location(0) Position: vec3<f32>;
@location(2) UV_0: vec2<f32>;
@location(1) Normal: vec3<f32>;
@location(3) Tangent: vec4<f32>;
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
@stage(vertex)
fn main(in: VertexIn, ) -> VertexOut {
var out:VertexOut;
var position = vec4<f32>( in.Position , 1.0 );
var normal = in.Normal;
out.v_uv = in.UV_0;
out.v_uv = out.v_uv * u_tilingOffset.xy + u_tilingOffset.zw;
out.v_normal = normalize(mat3x3<f32>(u_rendererData.u_normalMat[0].xyz, u_rendererData.u_normalMat[1].xyz, u_rendererData.u_normalMat[2].xyz) * normal);
var temp_pos = u_rendererData.u_modelMat * position;
out.v_pos = temp_pos.xyz / temp_pos.w;
out.position = u_rendererData.u_MVPMat * position;
return out;
}