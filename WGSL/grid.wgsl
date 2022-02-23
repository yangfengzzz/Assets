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
@group(0) @binding(14)
 var<uniform> u_cameraData: CameraData;
 @group(0) @binding(11)
 var<uniform> u_rendererData: RendererData;
 @group(0) @binding(13)
 var<uniform> u_tilingOffset: vec4<f32>;
 struct VertexIn {
@location(0) Position: vec3<f32>;
}
struct VertexOut {
@builtin(position) position: vec4<f32>;
@location(0) nearPoint: vec3<f32>;
@location(1) farPoint: vec3<f32>;
@location(2) fragView0: vec4<f32>;
@location(3) fragView1: vec4<f32>;
@location(4) fragView2: vec4<f32>;
@location(5) fragView3: vec4<f32>;
@location(6) fragProj0: vec4<f32>;
@location(7) fragProj1: vec4<f32>;
@location(8) fragProj2: vec4<f32>;
@location(9) fragProj3: vec4<f32>;
}
fn UnprojectPoint(x:f32, y:f32, z:f32, viewInv:mat4x4<f32>, projInv:mat4x4<f32>)->vec3<f32> {
    var unprojectedPoint =  viewInv * projInv * vec4<f32>(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}
@stage(vertex)
fn main(in: VertexIn, ) -> VertexOut {
var out:VertexOut;
out.nearPoint = UnprojectPoint(in.Position.x, in.Position.y, 0.0, u_cameraData.u_viewInvMat, u_cameraData.u_projInvMat).xyz;
out.farPoint = UnprojectPoint(in.Position.x, in.Position.y, 1.0, u_cameraData.u_viewInvMat, u_cameraData.u_projInvMat).xyz;
out.fragView0 = u_cameraData.u_viewMat[0];
out.fragView1 = u_cameraData.u_viewMat[1];
out.fragView2 = u_cameraData.u_viewMat[2];
out.fragView3 = u_cameraData.u_viewMat[3];
out.fragProj0 = u_cameraData.u_projMat[0];
out.fragProj1 = u_cameraData.u_projMat[1];
out.fragProj2 = u_cameraData.u_projMat[2];
out.fragProj3 = u_cameraData.u_projMat[3];
out.position = vec4<f32>(in.Position, 1.0);
return out;
}

struct Output {
@location(0) finalColor: vec4<f32>;
@builtin(frag_depth) depth: f32;
}
struct VertexOut {
@location(0) nearPoint: vec3<f32>;
@location(1) farPoint: vec3<f32>;
@location(2) fragView0: vec4<f32>;
@location(3) fragView1: vec4<f32>;
@location(4) fragView2: vec4<f32>;
@location(5) fragView3: vec4<f32>;
@location(6) fragProj0: vec4<f32>;
@location(7) fragProj1: vec4<f32>;
@location(8) fragProj2: vec4<f32>;
@location(9) fragProj3: vec4<f32>;
}
fn grid(fragPos3D:vec3<f32>, scale:f32, drawAxis:bool)->vec4<f32> {
   var coord = fragPos3D.xz * scale;
    var derivative = fwidth(coord);
    var grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    var line = min(grid.x, grid.y);
    var minimumz = min(derivative.y, 1.0);
    var minimumx = min(derivative.x, 1.0);
    var color = vec4<f32>(0.6, 0.6, 0.6, 1.0 - min(line, 1.0));
    if(fragPos3D.x > -1.0 * minimumx && fragPos3D.x < 1.0 * minimumx) {
        color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
    if(fragPos3D.z > -1.0 * minimumz && fragPos3D.z < 1.0 * minimumz) {
        color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    return color;
}
fn computeDepth(pos: vec3<f32>, fragView: mat4x4<f32>, fragProj: mat4x4<f32>)->f32 {
    var clip_space_pos = fragProj * fragView * vec4<f32>(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}
fn computeLinearDepth(pos: vec3<f32>, fragView: mat4x4<f32>, fragProj: mat4x4<f32>)->f32 {
    var near: f32 = 0.01;
    var far: f32 = 100.0;

    var clip_space_pos = fragProj * fragView * vec4<f32>(pos.xyz, 1.0);
    var clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0;
    var linearDepth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near));
    return linearDepth / far;
}
@stage(fragment)
fn main(in: VertexOut, ) -> Output {
var out:Output;
var t = -in.nearPoint.y / (in.farPoint.y - in.nearPoint.y);
var fragPos3D: vec3<f32> = in.nearPoint + t * (in.farPoint - in.nearPoint);

var fragView: mat4x4<f32> = mat4x4<f32>(in.fragView0, in.fragView1, in.fragView2, in.fragView3);
var fragProj: mat4x4<f32> = mat4x4<f32>(in.fragProj0, in.fragProj1, in.fragProj2, in.fragProj3);
var depth = computeDepth(fragPos3D, fragView, fragProj);

var linearDepth = computeLinearDepth(fragPos3D, fragView, fragProj);
var fading = max(0.0, (0.5 - linearDepth));
out.finalColor = (grid(fragPos3D, 1.0, true)) * f32(t > 0.0);
out.finalColor.a = out.finalColor.a * fading;
out.depth = depth;
return out;
}