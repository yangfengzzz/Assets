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
struct PointLight {
  color : vec3<f32>;
  position : vec3<f32>;
  distance : f32;
};
struct EnvMapLight {
  diffuse : vec3<f32>;
  mipMapLevel : f32;
  diffuseIntensity : f32;
  specularIntensity : f32;
};
struct PbrBaseData {
  baseColor : vec4<f32>;
  emissiveColor : vec4<f32>;
  normalTextureIntensity : f32;
  occlusionTextureIntensity : f32;
};
struct PbrData {
  metallic : f32;
  roughness : f32;
};
struct ReflectedLight {
    directDiffuse: vec3<f32>;
    directSpecular: vec3<f32>;
    indirectDiffuse: vec3<f32>;
    indirectSpecular: vec3<f32>;
};
struct GeometricContext {
    position: vec3<f32>;
    normal: vec3<f32>;
    viewDir: vec3<f32>;
};
struct PhysicalMaterial {
    diffuseColor: vec3<f32>;
    roughness: f32;
    specularColor: vec3<f32>;
    opacity: f32;
};
@group(0) @binding(11)
 var<uniform> u_cameraData: CameraData;
 @group(0) @binding(26)
 var<uniform> u_rendererData: RendererData;
 @group(0) @binding(0)
 var<uniform> u_pointLight: array<PointLight, 1>;
 @group(0) @binding(3)
 var<uniform> u_envMapLight: EnvMapLight;
 @group(0) @binding(12)
 var<uniform> u_alphaCutoff: f32;
 @group(0) @binding(14)
 var<uniform> u_pbrBaseData: PbrBaseData;
 @group(0) @binding(23)
 var<uniform> u_pbrData: PbrData;
 @group(0) @binding(15) var u_baseColorTexture: texture_2d<f32>;
 @group(0) @binding(16) var u_baseColorSampler: sampler;
 @group(0) @binding(17) var u_normalTexture: texture_2d<f32>;
 @group(0) @binding(18) var u_normalSampler: sampler;
 struct Output {
@location(0) finalColor: vec4<f32>;
}
struct VertexOut {
@location(0) v_uv: vec2<f32>;
@location(1) v_normalW: vec3<f32>;
@location(2) v_tangentW: vec3<f32>;
@location(3) v_bitangentW: vec3<f32>;
@location(4) v_pos: vec3<f32>;
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
fn getNormal(in:VertexOut, 
     u_normalTexture: texture_2d<f32>,
     u_normalSampler: sampler,
     normalIntensity: f32,
)->vec3<f32> {
var tbn =  mat3x3<f32>(in.v_tangentW, in.v_bitangentW, in.v_normalW );
var n = textureSample(u_normalTexture, u_normalSampler, in.v_uv ).rgb;
n = normalize(tbn * ((2.0 * n - 1.0) * vec3<f32>(normalIntensity, normalIntensity, 1.0)));
return n;
}
fn pow2(x: f32)->f32 {
    return x * x;
}
fn BRDF_Diffuse_Lambert(diffuseColor: vec3<f32>)->vec3<f32> {
    return RECIPROCAL_PI * diffuseColor;
}
fn computeSpecularOcclusion(ambientOcclusion: f32, roughness: f32, dotNV: f32)->f32 {
    return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}
fn getPhysicalMaterial(
     diffuseColor: vec4<f32>,
     metal: f32,
     roughness: f32,
     alphaCutoff: f32,
     v_uv: vec2<f32>,
     u_baseColorTexture: texture_2d<f32>,
     u_baseColorSampler: sampler,
    )-> PhysicalMaterial {
        var material: PhysicalMaterial;
var baseColor = textureSample(u_baseColorTexture, u_baseColorSampler, v_uv);
diffuseColor = diffuseColor * baseColor;
material.diffuseColor = diffuseColor.rgb * ( 1.0 - metal );
material.specularColor = mix( vec3<f32>( 0.04), diffuseColor.rgb, metal );
material.roughness = clamp( roughness, 0.04, 1.0 );
material.opacity = diffuseColor.a;
return material;
}
fn F_Schlick(specularColor: vec3<f32>, dotLH: f32)->vec3<f32> {
   var fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );
   return ( 1.0 - specularColor ) * fresnel + specularColor;
}
fn G_GGX_SmithCorrelated(alpha: f32, dotNL: f32, dotNV: f32)->f32 {
    var a2 = pow2( alpha );

    // dotNL and dotNV are explicitly swapped. This is not a mistake.
    var gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
    var gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );

    return 0.5 / max( gv + gl, EPSILON );
}
fn D_GGX(alpha: f32, dotNH: f32)->f32 {
   var a2 = pow2( alpha );

    var denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1

   return RECIPROCAL_PI * a2 / pow2( denom );
}
fn BRDF_Specular_GGX(incidentDirection:vec3<f32>, geometry:GeometricContext, specularColor:vec3<f32>, roughness:f32)->vec3<f32> {

    var alpha = pow2( roughness ); // UE4's roughness

    var halfDir = normalize( incidentDirection + geometry.viewDir );

    var dotNL = saturate( dot( geometry.normal, incidentDirection ) );
    var dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
    var dotNH = saturate( dot( geometry.normal, halfDir ) );
   var dotLH = saturate( dot( incidentDirection, halfDir ) );

    var F = F_Schlick( specularColor, dotLH );

   var G = G_GGX_SmithCorrelated( alpha, dotNL, dotNV );

   var D = D_GGX( alpha, dotNH );

   return F * ( G * D );

}
fn addDirectRadiance(incidentDirection:vec3<f32>, color:vec3<f32>, geometry:GeometricContext, 
material:PhysicalMaterial, reflectedLight:ptr<function, ReflectedLight>) {
   var dotNL = saturate( dot( geometry.normal, incidentDirection ) );

   var irradiance = dotNL * color;
   irradiance = irradiance * PI;

   (*reflectedLight).directSpecular = (*reflectedLight).directSpecular + irradiance * BRDF_Specular_GGX( incidentDirection, geometry, material.specularColor, material.roughness);

   (*reflectedLight).directDiffuse = (*reflectedLight).directDiffuse + irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );
}
fn addPointDirectLightRadiance(pointLight:PointLight, geometry:GeometricContext, material:PhysicalMaterial, reflectedLight:ptr<function, ReflectedLight>) {
    var lVector = pointLight.position - geometry.position;
    var direction = normalize( lVector );

    var lightDistance = length( lVector );

    var color = pointLight.color;
    color = color * clamp(1.0 - pow(lightDistance/pointLight.distance, 4.0), 0.0, 1.0);

    addDirectRadiance( direction, color, geometry, material, reflectedLight );
}
fn addTotalDirectRadiance(geometry:GeometricContext, material:PhysicalMaterial, reflectedLight:ptr<function, ReflectedLight>){
{
var i:i32 = 0;
loop {
if (i >= 1) { break; }
addPointDirectLightRadiance( u_pointLight[i], geometry, material, reflectedLight );
i = i + 1;
}
}
}
fn getLightProbeIrradiance(sh: array<vec3<f32>, 9>, normal:vec3<f32>)->vec3<f32> {
      var result = sh[0] +

            sh[1] * (normal.y) +
            sh[2] * (normal.z) +
            sh[3] * (normal.x) +

           sh[4] * (normal.y * normal.x) +
           sh[5] * (normal.y * normal.z) +
           sh[6] * (3.0 * normal.z * normal.z - 1.0) +
           sh[7] * (normal.z * normal.x) +
           sh[8] * (normal.x * normal.x - normal.y * normal.y);

   return max(result, vec3<f32>(0.0));
}
fn envBRDFApprox(specularColor:vec3<f32>, roughness:f32, dotNV:f32 )->vec3<f32>{
   let c0 = vec4<f32>( -1.0, -0.0275, -0.572, 0.022 );

   let c1 = vec4<f32>( 1.0, 0.0425, 1.04, -0.04 );

   var r = roughness * c0 + c1;

   var a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;

   var AB = vec2<f32>( -1.04, 1.04 ) * a004 + r.zw;

   return specularColor * AB.x + AB.y;
}
fn getSpecularMIPLevel(roughness:f32, maxMIPLevel:i32)->f32 {
    return roughness * f32(maxMIPLevel);
}
fn getLightProbeRadiance(geometry:GeometricContext, roughness:f32, maxMIPLevel:i32, specularIntensity:f32)->vec3<f32> {
return vec3<f32>(0.0, 0.0, 0.0);
}
@stage(fragment)
fn main(in: VertexOut, ) -> Output {
var out:Output;
var geometry = GeometricContext(in.v_pos, getNormal(in, 
u_normalTexture, u_normalSampler, u_pbrBaseData.normalTextureIntensity), normalize(u_cameraData.u_cameraPos - in.v_pos));
var material = getPhysicalMaterial(u_pbrBaseData.baseColor, u_pbrData.metallic, u_pbrData.roughness, u_alphaCutoff, 
in.v_uv, u_baseColorTexture, u_baseColorSampler,
);
var reflectedLight = ReflectedLight( vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0) );
var dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
addTotalDirectRadiance(geometry, material, &reflectedLight);
var irradiance = u_envMapLight.diffuse * u_envMapLight.diffuseIntensity;
irradiance = irradiance * PI;
reflectedLight.indirectDiffuse = reflectedLight.indirectDiffuse + irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );
var radiance = getLightProbeRadiance( geometry, material.roughness, i32(u_envMapLight.mipMapLevel), u_envMapLight.specularIntensity);
reflectedLight.indirectSpecular = reflectedLight.indirectSpecular + radiance * envBRDFApprox(material.specularColor, material.roughness, dotNV );
var emissiveRadiance = u_pbrBaseData.emissiveColor.rgb;
var totalRadiance =    reflectedLight.directDiffuse +
                        reflectedLight.indirectDiffuse +
                        reflectedLight.directSpecular +
                        reflectedLight.indirectSpecular +
                        emissiveRadiance;
out.finalColor =vec4<f32>(totalRadiance, material.opacity);
return out;
}
