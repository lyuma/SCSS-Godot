//shader_type spatial;

//---------------------------------------

// Keyword squeezing. 
//#if (_DETAIL_MULX2 || _DETAIL_MUL || _DETAIL_ADD || _DETAIL_LERP)
// const int _DETAIL = 1;
//#endif

//#if (_METALLICGLOSSMAP || _SPECGLOSSMAP)
// const int _SPECULAR = 1;
//#endif

//#if (_SUNDISK_NONE)
// const int _SUBSURFACE = 1;
//#endif

//---------------------------------------
uniform bool HEADER_SILENTS_CEL_SHADING_SHADER;
uniform sampler2D _MainTex : hint_albedo;
uniform vec4 _MainTex_ST = vec4(1.0,1.0,0.0,0.0);
uniform vec4 _Color = vec4(1.0,1.0,1.0,1.0);
uniform float _Cutoff: hint_range(0,1) = 0.5;
uniform bool _AlphaSharp = false;
uniform sampler2D _ColorMask : hint_white;
uniform sampler2D _ClippingMask : hint_white;
uniform sampler2D _BumpMap : hint_normal;
uniform float _BumpScale = 1.0;
uniform bool ENUM_COLOR_OUTLINECOLOR_ADDITIONALDATA;
uniform float _VertexColorType : hint_range(0,2,1) = 2;

uniform bool HEADER_EMISSION;
uniform sampler2D _EmissionMap : hint_albedo;
uniform vec3 _EmissionColor = vec3(0.0,0.0,0.0);
// For later use
const float _EmissionScrollX = 0.0;
const float _EmissionScrollY = 0.0;
const float _EmissionPhaseSpeed = 0.0;
const float _EmissionPhaseWidth = 0.0;

// CrossTone
uniform bool HEADER_CROSSTONE_SETTINGS;
//uniform bool ENABLE_THE_FOLLOWING_BOOLEAN_FOR_CROSSTONE;
uniform bool SCSS_CROSSTONE = true;
//#if defined(SCSS_CROSSTONE)

uniform sampler2D _1st_ShadeMap : hint_albedo;
uniform vec4 _1st_ShadeColor = vec4(0.0,0.0,0.0,1.0);

uniform sampler2D _2nd_ShadeMap : hint_albedo;
uniform vec4 _2nd_ShadeColor = vec4(0.0,0.0,0.0,1.0);

uniform float _1st_ShadeColor_Step : hint_range(0,1) = 0.5;
uniform float _1st_ShadeColor_Feather : hint_range(0.001, 1) = 0.01;
uniform float _2nd_ShadeColor_Step : hint_range(0,1) = 0.5;
uniform float _2nd_ShadeColor_Feather : hint_range(0.001, 1) = 0.01;

uniform sampler2D _ShadingGradeMap : hint_albedo;
uniform float _Tweak_ShadingGradeMapLevel : hint_range(-0.5, 0.5) = 0.0;

uniform bool ENUM_COMBINED_SEPARATE;
uniform float _CrosstoneToneSeparation : hint_range(0,1,1) = 0.0;
//#endif

uniform bool HEADERELSE_LIGHT_RAMP_SETTINGS;
//#if !defined(SCSS_CROSSTONE)
uniform bool ENUM_HORIZONTAL_VERTICAL_NONE;
uniform float _LightRampType : hint_range(0,2,1) = 0.0;
uniform sampler2D _Ramp : repeat_disable;
//uniform vec4 _Ramp_ST;
uniform bool ENUM_OCCLUSION_TONE_AUTO;
uniform float _ShadowMaskType : hint_range(0,2,1) = 0.0;
uniform sampler2D _ShadowMask;
// uniform vec4 _ShadowMask_ST;
uniform vec4 _ShadowMaskColor = vec4(1,1,1,1);
uniform float _Shadow : hint_range(0,1) = 0.5;
uniform float _ShadowLift : hint_range(-1,1) = 0.0;
uniform float _IndirectLightingBoost : hint_range(0,1) = 0.0;
//#endif

uniform bool HEADER_OUTLINE;
uniform bool ENUM_NONE_TINTED_COLORED;
uniform float _OutlineMode : hint_range(0,2,1) = 0.0;
uniform sampler2D _OutlineMask : hint_white;
uniform vec4 _OutlineMask_ST = vec4(1.0,1.0,0.0,0.0);
uniform float _outline_width = 0.1;
uniform vec4 _outline_color = vec4(0.5,0.5,0.5,1);
uniform float _UseInteriorOutline : hint_range(0,1,1) = 0.0;
uniform float _InteriorOutlineWidth : hint_range(0.0,1.0) = 0.01;

// uniform sampler2D _OutlineMask : hint_white; // unused if not outline shader.
// const float _outline_width = 0.0;
// const vec4 _outline_color = vec4(0.5,0.5,0.5,1.0);
// const float _OutlineMode = 0.0;
// const float _UseInteriorOutline = 0.0;
// const float _InteriorOutlineWidth = 0.01;

uniform bool HEADER_RIM;
uniform bool ENUM_DISABLE_LIT_AMBIENT_AMBIENTLIT;
uniform float _UseFresnel : hint_range(0,3,1);
uniform float _FresnelWidth : hint_range(0,20) = 0.5;
uniform float _FresnelStrength : hint_range(0.01,0.9999) = 0.5;
uniform vec4 _FresnelTint = vec4(1.0,1.0,1.0,1.0);
uniform float _UseFresnelLightMask : hint_range(0,1,1) = 0.0;
uniform float _FresnelLightMask : hint_range(0,1,1);
uniform vec4 _FresnelTintInv = vec4(1.0,1.0,1.0,1.0);
uniform float _FresnelWidthInv : hint_range(0,20) = 0.5;
uniform float _FresnelStrengthInv : hint_range(0.01, 0.9999) = 0.5;

uniform bool HEADER_SPECULAR;
//#if defined(_SPECULAR)
// uniform bool _SPECULAR = false;
uniform bool ENUM_DISABLE_STANDARD_CLOTH_ANISOTROPIC_CEL_CELSTRAND;
uniform float _SpecularType : hint_range(0,5,1);
uniform vec3 _SpecColor;
uniform sampler2D _SpecGlossMap : hint_white;
uniform vec4 _SpecGlossMap_ST = vec4(1.0,1.0,0.0,0.0);
uniform float _UseMetallic : hint_range(0,1,1) = 0.0;
uniform float _UseEnergyConservation : hint_range(0,1,1) = 0.0;
uniform float _Smoothness = 1.0;
uniform float _CelSpecularSoftness = 0.02;
uniform float _CelSpecularSteps = 1.0;
uniform float _Anisotropy = 0.8;
uniform bool _SPECULARHIGHLIGHTS = true;
///////uniform bool _GLOSSYREFLECTIONS = 1.0; // FIXME: Unity internal mumbo jumbo??
//#else
//#define _SpecularType 0
//#define _UseEnergyConservation 0
//uniform float _Anisotropy; // Can not be removed yet.
//#endif

/*
        		case SpecularType.Standard:
        		material.SetFloat("_SpecularType", 1);
        		material.EnableKeyword("_METALLICGLOSSMAP");
        		material.DisableKeyword("_SPECGLOSSMAP");
        		break;
        		case SpecularType.Cloth:
        		material.SetFloat("_SpecularType", 2);
        		material.EnableKeyword("_METALLICGLOSSMAP");
        		material.DisableKeyword("_SPECGLOSSMAP");
        		break;
        		case SpecularType.Anisotropic:
        		material.SetFloat("_SpecularType", 3);
        		material.EnableKeyword("_METALLICGLOSSMAP");
        		material.DisableKeyword("_SPECGLOSSMAP");
        		break;
        		case SpecularType.Cel:
        		material.SetFloat("_SpecularType", 4);
        		material.DisableKeyword("_METALLICGLOSSMAP");
        		material.EnableKeyword("_SPECGLOSSMAP");
        		break;
        		case SpecularType.CelStrand:
        		material.SetFloat("_SpecularType", 5);
        		material.DisableKeyword("_METALLICGLOSSMAP");
        		material.EnableKeyword("_SPECGLOSSMAP");
        		break;
        		case SpecularType.Disable:
        		material.SetFloat("_SpecularType", 0);
        		material.DisableKeyword("_METALLICGLOSSMAP");
        		material.DisableKeyword("_SPECGLOSSMAP");
        		break;
*/

bool _SPECULAR () {
	return _SpecularType != 0.0;
}

bool _METALLICGLOSSMAP () {
	return _SPECULAR () && _SpecularType <= 3.0;
}

bool _SPECGLOSSMAP () {
	return _SPECULAR () && !_METALLICGLOSSMAP();
}

uniform bool HEADER_MATCAP;
uniform bool ENUM_DISABLE_STANDARD_ANISOTROPIC;
uniform float _UseMatcap : hint_range(0,2,1) = 0.0;
uniform sampler2D _MatcapMask : hint_white;
// uniform vec4 _MatcapMask_ST; 
uniform sampler2D _Matcap1 : hint_black_albedo;
// uniform vec4 _Matcap1_ST; 
uniform float _Matcap1Strength : hint_range(0.0,2.0) = 1.0;
uniform bool ENUM_ADDITIVE_MULTIPLY_MEDIAN;
uniform int _Matcap1Blend : hint_range(0,2,1) = 0;
uniform sampler2D _Matcap2 : hint_black_albedo;
// uniform vec4 _Matcap2_ST; 
uniform float _Matcap2Strength : hint_range(0.0,2.0) = 1.0;
uniform bool ENUM_ADDITIVE_MULTIPLY_MEDIAN_;
uniform int _Matcap2Blend : hint_range(0,2,1) = 0;
uniform sampler2D _Matcap3 : hint_black_albedo;
// uniform vec4 _Matcap3_ST; 
uniform float _Matcap3Strength : hint_range(0.0,2.0) = 1.0;
uniform bool ENUM_ADDITIVE_MULTIPLY_MEDIAN__;
uniform int _Matcap3Blend : hint_range(0,2,1) = 0;
uniform sampler2D _Matcap4 : hint_black_albedo;
// uniform vec4 _Matcap4_ST; 
uniform float _Matcap4Strength : hint_range(0.0,2.0) = 1.0;
uniform bool ENUM_ADDITIVE_MULTIPLY_MEDIAN___;
uniform int _Matcap4Blend : hint_range(0,2,1) = 0;

uniform bool HEADER_DETAIL;
uniform bool _DETAIL = false;
const bool _DETAIL_MULX2 = true;
const bool _DETAIL_MUL = false;
const bool _DETAIL_ADD = false;
const bool _DETAIL_LERP = false;

//#if defined(_DETAIL)
uniform sampler2D _DetailAlbedoMap : hint_albedo;
uniform vec4 _DetailAlbedoMap_ST = vec4(1.0,1.0,0.0,0.0);
uniform float _DetailAlbedoMapScale = 1.0;
uniform sampler2D _DetailNormalMap : hint_normal;
uniform float _DetailNormalMapScale = 1.0;
uniform sampler2D _SpecularDetailMask : hint_white;
uniform float _SpecularDetailStrength = 1.0;
uniform sampler2D _DetailEmissionMap : hint_black_albedo;
uniform vec4 _EmissionDetailParams = vec4(0,0,0,0);
uniform float _UVSec : hint_range(0,1,1) = 0.0;
//#endif

//#if defined(_SUBSURFACE)
uniform bool HEADER_SUBSURFACE_SCATTERING;
uniform bool _SUBSURFACE = false;
// uniform vec4 _ThicknessMap_ST;
uniform sampler2D _ThicknessMap : hint_black;
uniform float _ThicknessMapInvert : hint_range(0,1,1) = 0.0;
uniform float _ThicknessMapPower : hint_range(0.01, 10.0) = 1.0;
uniform vec3 _SSSCol = vec3(1.0,1.0,1.0);
uniform float _SSSIntensity : hint_range(0.0,10.0) = 1.0;
uniform float _SSSPow : hint_range(0.01,10.0) = 1.0;
uniform float _SSSDist : hint_range(0,10) = 1.0;
uniform float _SSSAmbient : hint_range(0,1) = 0.0;
//#endif

// Animation
uniform bool HEADER_ANIMATION;
uniform bool _UseAnimation = false;//: hint_range(0,1,1) = 0.0;
uniform float _AnimationSpeed = 10.0;
uniform int _TotalFrames = 4;
uniform int _FrameNumber = 0;
uniform int _Columns = 2;
uniform int _Rows = 2;

// Vanishing
uniform bool HEADER_VANISHING;
uniform float _UseVanishing : hint_range(0,1,1) = 0.0;
uniform float _VanishingStart = 0.0;
uniform float _VanishingEnd = 0.0;

uniform bool HEADERALWAYS_OTHER;
uniform bool ENUM_TRANSPARENCY_SMOOTHNESS_CLIPPINGMASK;
uniform int _AlbedoAlphaMode : hint_range(0,2,1) = 0;
uniform vec4 _CustomFresnelColor = vec4(0.0,0.0,0.0,1.0);
uniform float _PixelSampleMode : hint_range(0,1,1) = 0.0;

uniform bool HEADERALWAYS_SYSTEM_LIGHTING;
uniform bool ENUM_DYNAMIC_DIRECTIONAL_FLATTEN;
uniform float _IndirectShadingType : hint_range(0,2,1) = 0.0;
uniform bool ENUM_ARKTOON_STANDARD_CUBED_DIRECTIONAL;
uniform float _LightingCalculationType : hint_range(0,3,1) = 0.0;
uniform vec4 _LightSkew = vec4(1.0, 0.1, 1.0, 0.0);
uniform float _DiffuseGeomShadowFactor : hint_range(0,1) = 1.0;
uniform float _LightWrappingCompensationFactor : hint_range(0.5, 1.0) = 0.8;

//-------------------------------------------------------------------------------------
// Input functions

//struct VertexOutput
//{
//    UNITY_VERTEX_INPUT_INSTANCE_ID
//    UNITY_VERTEX_OUTPUT_STEREO

//	UNITY_POSITION(pos);
//	vec3 normal : NORMAL;
//	vec4 color : COLOR0_centroid;
//	vec2 uv0 : TEXCOORD0;
//	vec2 uv1 : TEXCOORD1;
varying vec4 posWorld; // : TEXCOORD2;
//	vec3 normalDir : TEXCOORD3;
//	vec3 tangentDir : TEXCOORD4;
//	vec3 bitangentDir : TEXCOORD5;
//	vec4 vertex : VERTEX;

//	#if defined(VERTEXLIGHT_ON)
//	vec4 vertexLight : TEXCOORD6;
//	#endif

varying vec4 extraData; // : EXTRA_DATA;

	// Pass-through the shadow coordinates if this pass has shadows.
//	#if defined(USING_SHADOWS_UNITY)
//	UNITY_SHADOW_COORDS(8)
//	#endif

	// Pass-through the fog coordinates if this pass has fog.
//	#if defined(FOG_LINEAR) || defined(FOG_EXP) || defined(FOG_EXP2)
//	UNITY_FOG_COORDS(9)
//	#endif
//};

struct SCSS_RimLightInput
{
	float width;
	float power;
	vec3 tint;
	float alpha;

	float invWidth;
	float invPower;
	vec3 invTint;
	float invAlpha;
};

// Contains tonemap colour and shade offset.
struct SCSS_TonemapInput
{
	vec3 col; 
	float bias;
};

struct SCSS_Input 
{
	vec3 albedo;
	float alpha;
	vec3 normal;

	float occlusion;

	vec3 specColor;
	float oneMinusReflectivity;
	float smoothness;
	float perceptualRoughness;
	float softness;
	vec3 emission;

	vec3 thickness;

	SCSS_RimLightInput rim;
	SCSS_TonemapInput tone0;
	SCSS_TonemapInput tone1;

	vec3 specular_light;
};

struct SCSS_LightParam
{
	vec3 viewDir;
	vec3 halfDir;
	vec3 reflDir;
	vec2 rlPow4;
	float NdotL;
	float NdotV;
	float LdotH;
	float NdotH;
	float NdotAmb;
};





/////////////// TODO /////////////////
//#if defined(UNITY_STANDARD_BRDF_INCLUDED)
float getAmbientLight (vec3 normal, bool has_light, vec3 light_direction)
{
	vec3 ambientLightDirection = Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);

	if (_IndirectShadingType == 2.0) // Flatten
	{
		ambientLightDirection = has_light 
		? light_direction
		: ambientLightDirection;
	}

	float ambientLight = dot(normal, ambientLightDirection);
	ambientLight = ambientLight * 0.5 + 0.5;

	if (_IndirectShadingType == 0.0) // Dynamic
		ambientLight = getGreyscaleSH(normal);
	return ambientLight;
}

SCSS_LightParam initialiseLightParam (SCSS_Light l, vec3 normal, vec3 xposWorld, bool has_light, vec3 light_direction)
{
	SCSS_LightParam d;
	d.viewDir = normalize(l.cameraPos.xyz - xposWorld.xyz);
	d.halfDir = Unity_SafeNormalize (l.dir + d.viewDir);
	d.reflDir = reflect(-d.viewDir, normal); // Calculate reflection vector
	d.NdotL = (dot(l.dir, normal)); // Calculate NdotL
	d.NdotV = (dot(d.viewDir,  normal)); // Calculate NdotV
	d.LdotH = (dot(l.dir, d.halfDir));
	d.NdotH = (dot(normal, d.halfDir)); // Saturate seems to cause artifacts
	d.rlPow4 = Pow4_v2(vec2(dot(d.reflDir, l.dir), 1.0 - d.NdotV));
	d.NdotAmb = getAmbientLight(normal, has_light, light_direction);
	return d;
}
//#endif

// Allows saturate to be called on light params. 
// Does not affect directions. Those are already normalized.
// Only the required saturations will be left in code.
SCSS_LightParam saturatelp (SCSS_LightParam d)
{
	d.NdotL = saturate(d.NdotL);
	d.NdotV = saturate(d.NdotV);
	d.LdotH = saturate(d.LdotH);
	d.NdotH = saturate(d.NdotH);
	return d;
}

SCSS_RimLightInput initialiseRimParam()
{
	SCSS_RimLightInput rim;
	rim.width = _FresnelWidth;
	rim.power = _FresnelStrength;
	rim.tint = _FresnelTint.rgb;
	rim.alpha = _FresnelTint.a;

	rim.invWidth = _FresnelWidthInv;
	rim.invPower = _FresnelStrengthInv;
	rim.invTint = _FresnelTintInv.rgb;
	rim.invAlpha = _FresnelTintInv.a;
	return rim;
}

vec2 AnimateTexcoords(vec2 texcoord, float time)
{
	vec2 spriteUV = texcoord;
	if (_UseAnimation)
	{
		float fn = float(_FrameNumber) + floor(fract((time/20.0) * _AnimationSpeed) * float(_TotalFrames));

		float frame = clamp(fn, 0.0, float(_TotalFrames));

		vec2 offPerFrame = vec2((1.0 / float(_Columns)), (1.0 / float(_Rows)));

		vec2 spriteSize = texcoord * offPerFrame;

		vec2 currentSprite = 
				vec2(frame * offPerFrame.x,  1.0 - offPerFrame.y);
		
		float rowIndex;
		float mod = modf(frame / float(_Columns), rowIndex);
		currentSprite.y -= rowIndex * offPerFrame.y;
		currentSprite.x -= rowIndex * float(_Columns) * offPerFrame.x;
		
		spriteUV = (spriteSize + currentSprite); 
	}
	return spriteUV;
}

vec4 TexCoords(vec2 uv0, vec2 uv1)
{

    vec4 texcoord;
	texcoord.xy = TRANSFORM_TEX(uv0, _MainTex_ST);// Always source from uv0
	vec2 texSize = vec2(textureSize(_MainTex, 0).xy);
	texcoord.xy = (_PixelSampleMode != 0.0) ? 
		sharpSample(vec4(vec2(1.0)/texSize, texSize) * _MainTex_ST.xyxy, texcoord.xy) : texcoord.xy;

	if (_DETAIL) {
		texcoord.zw = TRANSFORM_TEX(((_UVSec == 0.0) ? uv0 : uv1), _DetailAlbedoMap_ST);
		texSize = vec2(textureSize(_DetailAlbedoMap, 0).xy);
		texcoord.zw = (_PixelSampleMode != 0.0) ? 
			sharpSample(vec4(vec2(1.0)/texSize, texSize) * _DetailAlbedoMap_ST.xyxy, texcoord.zw) : texcoord.zw;
	} else {
		texcoord.zw = texcoord.xy;
	}
    return texcoord;
}

//#define UNITY_SAMPLE_TEX2D_SAMPLER_LOD(tex,samplertex,coord,lod) tex.Sample (sampler##samplertex,coord,lod)
//#define UNITY_SAMPLE_TEX2D_LOD(tex,coord,lod) tex.Sample (sampler##tex,coord,lod)

float OutlineMask(vec2 uv)
{
	// Needs LOD, sampled in vertex function
    return textureLod(_OutlineMask, uv, 0.0).r;
}

float ColorMask(vec2 uv)
{
    return texture(_ColorMask, uv).g;
}

float RimMask(vec2 uv)
{
    return texture (_ColorMask, uv).b;
}

float DetailMask(vec2 uv)
{
    return texture (_ColorMask, uv).a;
}

vec4 MatcapMask(vec2 uv)
{
    return texture(_MatcapMask, uv);
}

vec3 Thickness(vec2 uv)
{
	if (_SUBSURFACE) {
		return pow(
			texture (_ThicknessMap, uv).rgb, 
			vec3(_ThicknessMapPower));
	} else {
		return vec3(1.0);
	}
}

const vec4 unity_ColorSpaceGrey = vec4(0.214041144, 0.214041144, 0.214041144, 0.5);
const vec4 unity_ColorSpaceDouble = vec4(4.59479380, 4.59479380, 4.59479380, 2.0);
const vec4 unity_ColorSpaceDielectricSpec = vec4(0.04, 0.04, 0.04, 1.0 - 0.04); // standard dielectric reflectivity coef at incident angle (= 4%)
const vec4 unity_ColorSpaceLuminance = vec4(0.0396819152, 0.458021790, 0.00609653955, 1.0); // Legacy: alpha is set to 1.0 to specify linear mode

vec3 Albedo(vec4 texcoords)
{
    vec3 albedo = texture (_MainTex, texcoords.xy).rgb * LerpWhiteTo(_Color.rgb, ColorMask(texcoords.xy));
	if (_DETAIL) {
		float mask = DetailMask(texcoords.xy);
		vec4 detailAlbedo = texture (_DetailAlbedoMap, texcoords.zw);
		mask *= detailAlbedo.a;
		mask *= _DetailAlbedoMapScale;
		if (_DETAIL_MULX2) {
			albedo *= LerpWhiteTo (detailAlbedo.rgb * unity_ColorSpaceDouble.rgb, mask);
		} else if (_DETAIL_MUL) {
			albedo *= LerpWhiteTo (detailAlbedo.rgb, mask);
		} else if (_DETAIL_ADD) {
			albedo += detailAlbedo.rgb * mask;
		} else if (_DETAIL_LERP) {
			albedo = mix (albedo, detailAlbedo.rgb, mask);
		}
	}
    return albedo;
}

float Alpha(vec2 uv)
{
	float alpha = _Color.a;
	switch(_AlbedoAlphaMode)
	{
		case 0: alpha *= texture(_MainTex, uv).a; break;
		case 2: alpha *= texture(_ClippingMask, uv).a; break;
	}
	return alpha;
}


vec4 SpecularGloss(vec4 texcoords, float mask)
{
    vec4 sg;
	if (textureSize(_SpecGlossMap, 0).x > 8) {
		sg = texture(_SpecGlossMap, texcoords.xy);

		sg.a = _AlbedoAlphaMode == 1? texture(_MainTex, texcoords.xy).a : sg.a;

		sg.rgb *= _SpecColor;
		sg.a *= _Smoothness; // _GlossMapScale is what Standard uses for this
	} else {
		sg.rgb = _SpecColor;
		sg.a = _AlbedoAlphaMode == 1? texture(_MainTex, texcoords.xy).a : sg.a;
	}

	if (_DETAIL) {
		vec4 sdm = texture(_SpecularDetailMask,texcoords.zw);
		sg *= saturate4(sdm + vec4(1.0-(_SpecularDetailStrength*mask)));
	}

    return sg;
}

vec3 Emission(vec2 uv)
{
    return texture(_EmissionMap, uv).rgb;
}

vec4 EmissionDetail(vec2 uv, float time)
{
	if (_DETAIL) {
		uv += _EmissionDetailParams.xy * time;
		vec4 ed = texture(_DetailEmissionMap, uv);
		ed.rgb = 
		_EmissionDetailParams.z > 0.0
		? (sin(ed.rgb * _EmissionDetailParams.w + vec3(time * _EmissionDetailParams.z)))+vec3(1.0) 
		: ed.rgb;
		return ed;
	} else {
		return vec4(1.0);
	}
}

vec3 NormalInTangentSpace(vec4 texcoords, float mask)
{
	vec3 normalTangent = UnpackScaleNormal(texture(_BumpMap, TRANSFORM_TEX(texcoords.xy, _MainTex_ST)), _BumpScale);
	if (_DETAIL) {
    	vec3 detailNormalTangent = UnpackScaleNormal(texture (_DetailNormalMap, texcoords.zw), _DetailNormalMapScale);
		if (_DETAIL_LERP) {
			normalTangent = mix(
				normalTangent,
				detailNormalTangent,
				mask);
		}else {
			normalTangent = mix(
				normalTangent,
				BlendNormalsPD(normalTangent, detailNormalTangent),
				mask);
		}
	}

    return normalTangent;
}

// This is based on a typical calculation for tonemapping
// scenes to screens, but in this case we want to flatten
// and shift the image colours.
// Lavender's the most aesthetic colour for this.
vec3 AutoToneMapping(vec3 color)
{
  	const float A = 0.7;
  	const vec3 B = vec3(.74, 0.6, .74); 
  	const float C = 0.0;
  	const float D = 1.59;
  	const float E = 0.451;
	color = max(vec3(0.0), color - vec3(0.004));
	color = (color * (A * color + B)) / (color * (C * color + D) + E);
	return color;
}

//#if !defined(SCSS_CROSSTONE)
SCSS_TonemapInput Tonemap(vec2 uv, inout float occlusion)
{
	SCSS_TonemapInput t;
	vec4 _ShadowMask_var = texture(_ShadowMask, uv.xy);

	// Occlusion
	if (_ShadowMaskType == 0.0)
	{
		// RGB will boost shadow range. Raising _Shadow reduces its influence.
		// Alpha will boost light range. Raising _Shadow reduces its influence.
		t.col = saturate(_IndirectLightingBoost+1.0-_ShadowMask_var.a) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.r;
	}
	// Tone
	if (_ShadowMaskType == 1.0)
	{
		t.col = saturate3(_ShadowMask_var.rgb+_IndirectLightingBoost) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.a;
	}
	// Auto-Tone
	if (_ShadowMaskType == 2.0)
	{
		vec3 albedo = Albedo(uv.xyxy);
		t.col = saturate3(AutoToneMapping(albedo)+_IndirectLightingBoost) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.r;
	}
	t.bias = (1.0 - _Shadow) * t.bias + _Shadow;
	occlusion = t.bias;
	return t;
}

// Sample ramp with the specified options.
// rampPosition: 0-1 position on the light ramp from light to dark
// softness: 0-1 position on the light ramp on the other axis
vec3 sampleRampWithOptions(float rampPosition, float softness) 
{
	if (_LightRampType == 3.0) // No sampling
	{
		return vec3(saturate(rampPosition*2.0-1.0));
	}
	if (_LightRampType == 2.0) // None
	{
		float shadeWidth = 0.0002 * (1.0+softness*100.0);

		const float shadeOffset = 0.5; 
		float lightContribution = simpleSharpen(rampPosition, shadeWidth, shadeOffset, 2);
		return vec3(saturate(lightContribution));
	}
	if (_LightRampType == 1.0) // Vertical
	{
		vec2 rampUV = vec2(softness, rampPosition);
		return texture(_Ramp, saturate2(rampUV)).rgb;
	}
	else // Horizontal
	{
		vec2 rampUV = vec2(rampPosition, softness);
		return texture(_Ramp, saturate2(rampUV)).rgb;
	}
}
//#endif

//#if defined(SCSS_CROSSTONE)
// Tonemaps contain tone in RGB, occlusion in A.
// Midpoint/width is handled in the application function.
SCSS_TonemapInput Tonemap1st (vec2 uv)
{
	vec4 tonemap = texture(_1st_ShadeMap, uv.xy);
	tonemap.rgb = tonemap.rgb * _1st_ShadeColor.rgb;
	SCSS_TonemapInput t;
	t.col = tonemap.rgb;
	t.bias = tonemap.a;
	return t;
}
SCSS_TonemapInput Tonemap2nd (vec2 uv)
{
	vec4 tonemap = texture(_2nd_ShadeMap, uv.xy);
	tonemap.rgb *= _2nd_ShadeColor.rgb;
	SCSS_TonemapInput t;
	t.col = tonemap.rgb;
	t.bias = tonemap.a;
	return t;
}

float adjustShadeMap(float x, float y)
{
	// Might be changed later.
	return (x * (1.0+y));

}

float ShadingGradeMap (vec2 uv)
{
	vec4 tonemap = texture(_ShadingGradeMap, uv.xy);
	return adjustShadeMap(tonemap.g, _Tweak_ShadingGradeMapLevel);
}
//#endif

// float innerOutline (VertexOutput i)
// {
// 	// The compiler should merge this with the later calls.
// 	// Use the vertex normals for this to avoid artifacts.
// 	SCSS_LightParam d = initialiseLightParam((SCSS_Light)0, i.normalDir, i.posWorld.xyz);
// 	float baseRim = d.NdotV;
// 	baseRim = simpleSharpen(baseRim, 0, _InteriorOutlineWidth * OutlineMask(i.uv0.xy));
// 	return baseRim;
// }

vec3 applyOutlineV3(vec3 col, float is_outline)
{    
	col = mix(col, col * _outline_color.rgb, is_outline);
    if (_OutlineMode == 2.0) 
    {
        col = mix(col, _outline_color.rgb, is_outline);
    }
    return col;
}

SCSS_Input applyOutline(SCSS_Input c, float is_outline)
{

	c.albedo = applyOutlineV3(c.albedo, is_outline);
    if (_CrosstoneToneSeparation == 1.0) 
    {
        c.tone0.col = applyOutlineV3(c.tone0.col, is_outline);
        c.tone1.col = applyOutlineV3(c.tone1.col, is_outline);
    }

    return c;
}

void applyVanishing (vec3 baseCameraPos, vec3 baseWorldPos, inout float alpha) {
    float closeDist = distance(baseCameraPos, baseWorldPos);
    float vanishing = saturate(lerpstep(_VanishingStart, _VanishingEnd, closeDist));
    alpha = mix(alpha, alpha * vanishing, _UseVanishing);
}

































































































































































































































































