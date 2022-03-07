struct ProjectionUniforms {
  matrix : mat4x4<f32>;
  inverseMatrix : mat4x4<f32>;
  outputSize : vec2<f32>;
  zNear : f32;
  zFar : f32;
};
struct ClusterBounds {
  minAABB : vec3<f32>;
  maxAABB : vec3<f32>;
};
struct Clusters {
  bounds : array<ClusterBounds, 27648>;
};
@group(0) @binding(11)
 var<uniform> u_cluster_projection: ProjectionUniforms;
 @group(0) @binding(13)
 var<storage, read_write> u_clusters: Clusters;
 fn lineIntersectionToZPlane(a : vec3<f32>, b : vec3<f32>, zDistance : f32) -> vec3<f32> {
  let normal = vec3<f32>(0.0, 0.0, 1.0);
  let ab = b - a;
  let t = (zDistance - dot(normal, a)) / dot(normal, ab);
  return a + t * ab;
}

fn clipToView(clip : vec4<f32>) -> vec4<f32> {
  let view = u_cluster_projection.inverseMatrix * clip;
  return view / vec4<f32>(view.w, view.w, view.w, view.w);
}

fn screen2View(screen : vec4<f32>) -> vec4<f32> {
  let texCoord = screen.xy / u_cluster_projection.outputSize.xy;
  let clip = vec4<f32>(vec2<f32>(texCoord.x, 1.0 - texCoord.y) * 2.0 - vec2<f32>(1.0, 1.0), screen.z, screen.w);
  return clipToView(clip);
}
let tileCount = vec3<u32>(32u, 18u, 48u);
let eyePos = vec3<f32>(0.0, 0.0, 0.0);
@stage(compute) @workgroup_size(4, 2, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
) {
let tileIndex = global_id.x +
    global_id.y * tileCount.x +
    global_id.z * tileCount.x * tileCount.y;

let tileSize = vec2<f32>(u_cluster_projection.outputSize.x / f32(tileCount.x),
u_cluster_projection.outputSize.y / f32(tileCount.y));

let maxPoint_sS = vec4<f32>(vec2<f32>(f32(global_id.x+1u), f32(global_id.y+1u)) * tileSize, 0.0, 1.0);
let minPoint_sS = vec4<f32>(vec2<f32>(f32(global_id.x), f32(global_id.y)) * tileSize, 0.0, 1.0);

let maxPoint_vS = screen2View(maxPoint_sS).xyz;
let minPoint_vS = screen2View(minPoint_sS).xyz;

let tileNear = -u_cluster_projection.zNear * pow(u_cluster_projection.zFar/ u_cluster_projection.zNear, f32(global_id.z)/f32(tileCount.z));
let tileFar = -u_cluster_projection.zNear * pow(u_cluster_projection.zFar/ u_cluster_projection.zNear, f32(global_id.z+1u)/f32(tileCount.z));

let minPointNear = lineIntersectionToZPlane(eyePos, minPoint_vS, tileNear);
let minPointFar = lineIntersectionToZPlane(eyePos, minPoint_vS, tileFar);
let maxPointNear = lineIntersectionToZPlane(eyePos, maxPoint_vS, tileNear);
let maxPointFar = lineIntersectionToZPlane(eyePos, maxPoint_vS, tileFar);

u_clusters.bounds[tileIndex].minAABB = min(min(minPointNear, minPointFar),min(maxPointNear, maxPointFar));
u_clusters.bounds[tileIndex].maxAABB = max(max(minPointNear, minPointFar),max(maxPointNear, maxPointFar));
}