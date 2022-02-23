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
 @group(0) @binding(12)
 var<uniform> u_rendererData: RendererData;
 @group(0) @binding(17)
 var<uniform> u_tilingOffset: vec4<f32>;
 struct VertexOut {
@location(0) v_uv: vec2<f32>;
@builtin(position) position: vec4<f32>;
}
struct VertexIn {
@location(0) Position: vec3<f32>;
@location(2) UV_0: vec2<f32>;
}
@stage(vertex)
fn main(in: VertexIn, ) -> VertexOut {
var out:VertexOut;
var position = vec4<f32>( in.Position , 1.0 );
out.v_uv = in.UV_0;
out.v_uv = out.v_uv * u_tilingOffset.xy + u_tilingOffset.zw;
out.position = u_rendererData.u_MVPMat * position;
return out;
}