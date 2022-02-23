let offsets = array<vec2<f32>, 4>(
    vec2<f32>(0, 0),
    vec2<f32>(0.5, 0),
    vec2<f32>(0, 0.5),
    vec2<f32>(0.5, 0.5)
);

struct ShadowData {
     bias:f32;
     intensity:f32;
     radius:f32;
     dump:f32;
     vp:array<mat4x4<f32>, 4>;
     cascadeSplits:array<f32, 4>;
};

// @group(0) @binding(12)
// var<uniform> u_shadowData: array<ShadowData, 1>;
// @group(0) @binding(15) var u_shadowMap: texture_depth_2d_array;
// @group(0) @binding(16) var u_shadowSampler: sampler_comparison;

fn textureProj( worldPos:vec3<f32>,  viewPos:vec3<f32>,  off:vec2<f32>,
                   index:i32)->f32 {
    // Get cascade index for the current fragment's view position
    var cascadeIndex:i32 = 0;
    var scale:f32 = 1.0;
    if (u_shadowData[index].cascadeSplits[0] * u_shadowData[index].cascadeSplits[1] > 0.0) {
        scale = 0.5;
        for(var i:i32 = 0; i < 4 - 1; i = i+1) {
            if(viewPos.z < u_shadowData[index].cascadeSplits[i]) {
                cascadeIndex = i + 1;
            }
        }
    }

    var shadowCoord:vec4<f32> = u_shadowData[index].vp[cascadeIndex] * vec4<f32>(worldPos, 1.0);
    var xy = shadowCoord.xy;
    xy /= shadowCoord.w;
    xy = xy * 0.5 + 0.5;
    xy.y = 1.0 - xy.y;
    xy *= scale;
    var shadow_sample = textureSampleCompare(u_shadowMap, u_shadowSampler, xy + off + offsets[cascadeIndex], index, shadowCoord.z / shadowCoord.w);
    return shadow_sample * u_shadowData[index].intensity;
}

fn filterPCF( worldPos:vec3<f32>,  viewPos:vec3<f32>,
                 index:i32)->f32 {
    // Get cascade index for the current fragment's view position
    var cascadeIndex = 0;
    var scale = 1.0;
    if (u_shadowData[index].cascadeSplits[0] * u_shadowData[index].cascadeSplits[1] > 0.0) {
        scale = 0.5;
        for(var i = 0; i < 4 - 1; i = i + 1) {
            if(viewPos.z < u_shadowData[index].cascadeSplits[i]) {
                cascadeIndex = i + 1;
            }
        }
    }
    
    var shadowCoord = u_shadowData[index].vp[cascadeIndex] * vec4<f32>(worldPos, 1.0);
    var xy = shadowCoord.xy;
    xy /= shadowCoord.w;
    xy = xy * 0.5 + 0.5;
    xy.y = 1.0 - xy.y;
    xy *= scale;
    
    let neighborWidth = 3.0;
    let neighbors = (neighborWidth * 2.0 + 1.0) * (neighborWidth * 2.0 + 1.0);
    let mapSize = 4096.0;
    let texelSize = 1.0 / mapSize;
    var total = 0.0;
    for (var x = -neighborWidth; x <= neighborWidth; x = x + 1.0) {
        for (var y = -neighborWidth; y <= neighborWidth; y = y + 1.0) {
            var shadow_sample = textureSampleCompare(u_shadowMap, u_shadowSampler, 
                                                    xy + vec2<f32>(x, y) * texelSize + offsets[cascadeIndex], 
                                                    index, shadowCoord.z / shadowCoord.w);
            total += shadow_sample * u_shadowData[index].intensity;
        }
    }
    return total / neighbors;
}

