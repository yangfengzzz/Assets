let PI:f32 = 3.14159265359;
let RECIPROCAL_PI:f32 = 0.31830988618;
let EPSILON:f32 = 1.0e-6;
let LOG2:f32 = 1.442695;
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
@group(0) @binding(27)
 var<uniform> u_jointMatrix: array<mat4x4<f32>, 19>;
 @group(0) @binding(11)
 var<uniform> u_cameraData: CameraData;
 @group(0) @binding(26)
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
@location(7) Joints_0: vec4<f32>;
@location(6) Weights_0: vec4<f32>;
@location(1) Normal: vec3<f32>;
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
@stage(vertex)
fn main(in: VertexIn, ) -> VertexOut {
var out:VertexOut;
var position = vec4<f32>( in.Position , 1.0 );
var normal = in.Normal;
var skinMatrix = in.Weights_0.x * u_jointMatrix[ i32( in.Joints_0.x ) ] +
in.Weights_0.y * u_jointMatrix[ i32( in.Joints_0.y ) ] +
in.Weights_0.z * u_jointMatrix[ i32( in.Joints_0.z ) ] +
in.Weights_0.w * u_jointMatrix[ i32( in.Joints_0.w ) ];
position = skinMatrix * position;
normal = vec4<f32>( skinMatrix * vec4<f32>( normal, 0.0 ) ).xyz;
out.v_uv = in.UV_0;
out.v_uv = out.v_uv * u_tilingOffset.xy + u_tilingOffset.zw;
out.v_normal = normalize(mat3x3<f32>(u_rendererData.u_normalMat[0].xyz, u_rendererData.u_normalMat[1].xyz, u_rendererData.u_normalMat[2].xyz) * normal);
var temp_pos = u_rendererData.u_modelMat * position;
out.v_pos = temp_pos.xyz / temp_pos.w;
out.position = u_rendererData.u_MVPMat * position;
return out;
}