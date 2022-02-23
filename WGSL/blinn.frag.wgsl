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

struct DirectLight {
  color : vec3<f32>;
  direction : vec3<f32>;
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

struct BlinnPhongData {
  baseColor : vec4<f32>;
  specularColor : vec4<f32>;
  emissiveColor : vec4<f32>;
  normalIntensity : f32;
  shininess : f32;
};

@group(0) @binding(11)
 var<uniform> u_cameraData: CameraData;
 @group(0) @binding(23)
 var<uniform> u_rendererData: RendererData;
 @group(0) @binding(2)
 var<uniform> u_directLight: array<DirectLight, 1>;
 @group(0) @binding(1)
 var<uniform> u_spotLight: array<SpotLight, 1>;
 @group(0) @binding(3)
 var<uniform> u_envMapLight: EnvMapLight;
 @group(0) @binding(14)
 var<uniform> u_blinnPhongData: BlinnPhongData;
 @group(0) @binding(12)
 var<uniform> u_alphaCutoff: f32;
 @group(0) @binding(15) var u_diffuseTexture: texture_2d<f32>;
 @group(0) @binding(16) var u_diffuseSampler: sampler;
 struct Output {
@location(0) finalColor: vec4<f32>;
}
struct VertexOut {
@location(0) v_uv: vec2<f32>;
@location(1) v_normal: vec3<f32>;
@location(2) v_pos: vec3<f32>;
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
fn getNormal(in:VertexOut)->vec3<f32> {
var n = normalize(in.v_normal);
return n;
}
@stage(fragment)
fn main(in: VertexOut, ) -> Output {
var out:Output;
var ambient = vec4<f32>(0.0);
var emission = u_blinnPhongData.emissiveColor;
var diffuse = u_blinnPhongData.baseColor;
var specular = u_blinnPhongData.specularColor;
var diffuseTextureColor = textureSample(u_diffuseTexture, u_diffuseSampler, in.v_uv);
diffuse = diffuse * diffuseTextureColor;
ambient = vec4<f32>(u_envMapLight.diffuse * u_envMapLight.diffuseIntensity, 1.0) * diffuse;
var V = normalize(u_cameraData.u_cameraPos - in.v_pos);
var N = getNormal(in);
var lightDiffuse = vec3<f32>( 0.0, 0.0, 0.0 );
var lightSpecular = vec3<f32>( 0.0, 0.0, 0.0 );
{
var i:i32 = 0;
loop {
if (i >= 1) { break; }
    var d:f32 = max(dot(N, -u_directLight[i].direction), 0.0);
    lightDiffuse = lightDiffuse + u_directLight[i].color * d;

    var halfDir:vec3<f32> = normalize( V - u_directLight[i].direction );
    var s:f32 = pow( clamp( dot( N, halfDir ), 0.0, 1.0 ), u_blinnPhongData.shininess );
    lightSpecular = lightSpecular + u_directLight[i].color * s;
i = i + 1;
}
}
{
var i:i32 = 0;
loop {
if (i >= 1) { break; }
    var direction = u_spotLight[i].position - in.v_pos;
    var lightDistance = length( direction );
    direction = direction / lightDistance;
    var angleCos = dot( direction, -u_spotLight[i].direction );
    var decay = clamp(1.0 - pow(lightDistance/u_spotLight[i].distance, 4.0), 0.0, 1.0);
    var spotEffect = smoothStep( u_spotLight[i].penumbraCos, u_spotLight[i].angleCos, angleCos );
    var decayTotal = decay * spotEffect;
    var d = max( dot( N, direction ), 0.0 )  * decayTotal;
    lightDiffuse = lightDiffuse + u_spotLight[i].color * d;

    var halfDir = normalize( V + direction );
    var s = pow( clamp( dot( N, halfDir ), 0.0, 1.0 ), u_blinnPhongData.shininess ) * decayTotal;
    lightSpecular = lightSpecular + u_spotLight[i].color * s;
i = i + 1;
}
}
diffuse = diffuse * vec4<f32>(lightDiffuse, 1.0);
specular = specular * vec4<f32>(lightSpecular, 1.0);
out.finalColor = emission + ambient + diffuse + specular;
out.finalColor.a = diffuse.a;
return out;
}
