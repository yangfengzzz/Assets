struct ProjectionUniforms {
  matrix : mat4x4<f32>;
  inverseMatrix : mat4x4<f32>;
  outputSize : vec2<f32>;
  zNear : f32;
  zFar : f32;
};
struct ViewUniforms {
  matrix : mat4x4<f32>;
  position : vec3<f32>;
};
struct SpotLight {
  color : vec3<f32>;
  position : vec3<f32>;
  direction : vec3<f32>;
  distance : f32;
  angleCos : f32;
  penumbraCos : f32;
};
struct EnvMapLight {
  diffuse : vec3<f32>;
  mipMapLevel : f32;
  diffuseIntensity : f32;
  specularIntensity : f32;
};
struct ClusterLights {
  offset : u32;
  count : u32;
};
struct ClusterLightGroup {
  offset : atomic<u32>;
  lights : array<ClusterLights, 27648>;
  indices : array<u32, 2764800>;
};
struct ClusterBounds {
  minAABB : vec3<f32>;
  maxAABB : vec3<f32>;
};
struct Clusters {
  bounds : array<ClusterBounds, 27648>;
};
let tileCount : vec3<u32> = vec3<u32>(32u, 18u, 48);
@group(0) @binding(11)
 var<uniform> u_cluster_projection: ProjectionUniforms;
 @group(0) @binding(12)
 var<uniform> u_cluster_view: ViewUniforms;
 @group(0) @binding(9)
 var<uniform> u_spotLight: array<SpotLight, 1>;
 @group(0) @binding(0)
 var<uniform> u_envMapLight: EnvMapLight;
 @group(0) @binding(14)
 var<storage, read_write> u_clusterLights: ClusterLightGroup;
 @group(0) @binding(13)
 var<storage, read_write> u_clusters: Clusters;
 fn linearDepth(depthSample : f32) -> f32 {
  return u_cluster_projection.zFar * u_cluster_projection.zNear / fma(depthSample, u_cluster_projection.zNear - u_cluster_projection.zFar, u_cluster_projection.zFar);
}

fn getTile(fragCoord : vec4<f32>) -> vec3<u32> {
  // TODO: scale and bias calculation can be moved outside the shader to save cycles.
  let sliceScale = f32(tileCount.z) / log2(u_cluster_projection.zFar / u_cluster_projection.zNear);
  let sliceBias = -(f32(tileCount.z) * log2(u_cluster_projection.zNear) / log2(u_cluster_projection.zFar / u_cluster_projection.zNear));
  let zTile = u32(max(log2(linearDepth(fragCoord.z)) * sliceScale + sliceBias, 0.0));

  return vec3<u32>(u32(fragCoord.x / (u_cluster_projection.outputSize.x / f32(tileCount.x))),
      u32(fragCoord.y / (u_cluster_projection.outputSize.y / f32(tileCount.y))),
      zTile);
}

fn getClusterIndex(fragCoord : vec4<f32>) -> u32 {
  let tile = getTile(fragCoord);
  return tile.x +
      tile.y * tileCount.x +
      tile.z * tileCount.x * tileCount.y;
}
fn sqDistPointAABB(point : vec3<f32>, minAABB : vec3<f32>, maxAABB : vec3<f32>) -> f32 {
  var sqDist = 0.0;
  // const minAABB : vec3<f32> = u_clusters.bounds[tileIndex].minAABB;
  // const maxAABB : vec3<f32> = u_clusters.bounds[tileIndex].maxAABB;

  // Wait, does this actually work? Just porting code, but it seems suspect?
  for(var i = 0; i < 3; i = i + 1) {
    let v = point[i];
    if(v < minAABB[i]){
      sqDist = sqDist + (minAABB[i] - v) * (minAABB[i] - v);
    }
    if(v > maxAABB[i]){
      sqDist = sqDist + (v - maxAABB[i]) * (v - maxAABB[i]);
    }
  }

  return sqDist;
}
@stage(compute) @workgroup_size(4, 2, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
) {
let tileIndex = global_id.x +
    global_id.y * tileCount.x +
    global_id.z * tileCount.x * tileCount.y;

var clusterLightCount = 0u;
var cluserLightIndices : array<u32, 100>;
for (var i = 0u; i < 1u; i = i + 1u) {
  let range = u_spotLight[i].distance;
  // Lights without an explicit range affect every cluster, but this is a poor way to handle that.
  var lightInCluster = range <= 0.0;

  if (!lightInCluster) {
    let lightViewPos = u_cluster_view.matrix * vec4<f32>(u_spotLight[i].position, 1.0);
    let sqDist = sqDistPointAABB(lightViewPos.xyz, u_clusters.bounds[tileIndex].minAABB, u_clusters.bounds[tileIndex].maxAABB);
    lightInCluster = sqDist <= (range * range);
  }

  if (lightInCluster) {
    // Light affects this cluster. Add it to the list.
    cluserLightIndices[clusterLightCount] = i;
    clusterLightCount = clusterLightCount + 1u;
  }

  if (clusterLightCount == 100u) {
    break;
  }
}

var offset = atomicAdd(&u_clusterLights.offset, clusterLightCount);

for(var i = 0u; i < clusterLightCount; i = i + 1u) {
  u_clusterLights.indices[offset + i] = cluserLightIndices[i];
}
u_clusterLights.lights[tileIndex].offset = offset;
u_clusterLights.lights[tileIndex].count = clusterLightCount;
}