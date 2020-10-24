shader_type spatial;

//---------------------------------------

// Keyword squeezing. 
//#if (_DETAIL_MULX2 || _DETAIL_MUL || _DETAIL_ADD || _DETAIL_LERP)
const int _DETAIL = 1;
//#endif

//#if (_METALLICGLOSSMAP || _SPECGLOSSMAP)
const int _SPECULAR = 1;
//#endif

//#if (_SUNDISK_NONE)
const int _SUBSURFACE = 1;
//#endif

//---------------------------------------

uniform sampler2D _MainTex : hint_albedo;
uniform vec4 _MainTex_ST;
uniform sampler2D _ColorMask;
uniform sampler2D _ClippingMask;
uniform sampler2D _BumpMap : hint_normal;
uniform sampler2D _EmissionMap : hint_albedo_black;

//#if defined(_DETAIL)
uniform sampler2D _DetailAlbedoMap;
uniform vec4 _DetailAlbedoMap_ST;
uniform sampler2D _DetailNormalMap;
uniform sampler2D _DetailEmissionMap
uniform sampler2D _SpecularDetailMask;

uniform float _DetailAlbedoMapScale;
uniform float _DetailNormalMapScale;
uniform float _SpecularDetailStrength;
//#endif

//#if defined(_SPECULAR)
//uniform vec4 _SpecColor; // Defined elsewhere
sampler2D _SpecGlossMap;
uniform vec4 _SpecGlossMap_ST;
uniform float _UseMetallic;
uniform float _SpecularType;
uniform float _Smoothness;
uniform float _UseEnergyConservation;
uniform float _Anisotropy;
uniform float _CelSpecularSoftness;
uniform float _CelSpecularSteps;
//#else
//#define _SpecularType 0
//#define _UseEnergyConservation 0
//uniform float _Anisotropy; // Can not be removed yet.
//#endif

//#if defined(SCSS_CROSSTONE)
uniform sampler2D _1st_ShadeMap : hint_albedo;
uniform sampler2D _2nd_ShadeMap : hint_albedo;
uniform sampler2D _ShadingGradeMap : hint_albedo;
//#endif

uniform float _Shadow;
uniform float _ShadowLift;

//#if !defined(SCSS_CROSSTONE)
uniform sampler2D _ShadowMask;
uniform vec4 _ShadowMask_ST;
uniform sampler2D _Ramp;
uniform vec4 _Ramp_ST;
uniform float _LightRampType;
uniform vec4 _ShadowMaskColor;
uniform float _ShadowMaskType;
uniform float _IndirectLightingBoost;
//#endif

uniform sampler2D _MatcapMask;
uniform vec4 _MatcapMask_ST; 
uniform sampler2D _Matcap1;
uniform vec4 _Matcap1_ST; 
uniform sampler2D _Matcap2;
uniform vec4 _Matcap2_ST; 
uniform sampler2D _Matcap3;
uniform vec4 _Matcap3_ST; 
uniform sampler2D _Matcap4;
uniform vec4 _Matcap4_ST; 

uniform vec4 _Color;
uniform float _BumpScale;
uniform float _Cutoff;
uniform float _AlphaSharp;
uniform float _UVSec;
uniform float _AlbedoAlphaMode;

uniform vec4 _EmissionColor;
uniform vec4 _EmissionDetailParams;
// For later use
uniform float _EmissionScrollX;
uniform float _EmissionScrollY;
uniform float _EmissionPhaseSpeed;
uniform float _EmissionPhaseWidth;

uniform float _UseFresnel;
uniform float _UseFresnelLightMask;
uniform vec4 _FresnelTint;
uniform float _FresnelWidth;
uniform float _FresnelStrength;
uniform float _FresnelLightMask;
uniform vec4 _FresnelTintInv;
uniform float _FresnelWidthInv;
uniform float _FresnelStrengthInv;

uniform vec4 _CustomFresnelColor;

uniform float _outline_width;
uniform vec4 _outline_color;
uniform float _OutlineMode;

uniform float _LightingCalculationType;

uniform float _UseMatcap;
uniform float _Matcap1Strength;
uniform float _Matcap2Strength;
uniform float _Matcap3Strength;
uniform float _Matcap4Strength;
uniform float _Matcap1Blend;
uniform float _Matcap2Blend;
uniform float _Matcap3Blend;
uniform float _Matcap4Blend;

//#if defined(_SUBSURFACE)
uniform sampler2D _ThicknessMap;
uniform vec4 _ThicknessMap_ST;
uniform float _UseSubsurfaceScattering;
uniform float _ThicknessMapPower;
uniform float _ThicknessMapInvert;
uniform vec3 _SSSCol;
uniform float _SSSIntensity;
uniform float _SSSPow;
uniform float _SSSDist;
uniform float _SSSAmbient;
//#endif

uniform vec4 _LightSkew;
uniform float _PixelSampleMode;
uniform float _VertexColorType;

// CrossTone
uniform vec4 _1st_ShadeColor;
uniform vec4 _2nd_ShadeColor;
uniform float _1st_ShadeColor_Step;
uniform float _1st_ShadeColor_Feather;
uniform float _2nd_ShadeColor_Step;
uniform float _2nd_ShadeColor_Feather;

uniform float _Tweak_ShadingGradeMapLevel;

uniform float _DiffuseGeomShadowFactor;
uniform float _LightWrappingCompensationFactor;

uniform float _IndirectShadingType;
uniform float _CrosstoneToneSeparation;

uniform float _UseInteriorOutline;
uniform float _InteriorOutlineWidth;

uniform sampler2D _OutlineMask;
uniform vec4 _OutlineMask_ST; 

// Animation
uniform float _UseAnimation;
uniform float _AnimationSpeed;
uniform int _TotalFrames;
uniform int _FrameNumber;
uniform int _Columns;
uniform int _Rows;

// Vanishing
uniform float _UseVanishing;
uniform float _VanishingStart;
uniform float _VanishingEnd;

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
	float oneMinusReflectivity, smoothness, perceptualRoughness;
	float softness;
	vec3 emission;

	vec3 thickness;

	SCSS_RimLightInput rim;
	SCSS_TonemapInput tone[2];
};

struct SCSS_LightParam
{
	vec3 viewDir, halfDir, reflDir;
	vec2 rlPow4;
	float NdotL, NdotV, LdotH, NdotH;
	float NdotAmb;
};

/////////////// TODO /////////////////
/*
#if defined(UNITY_STANDARD_BRDF_INCLUDED)
float getAmbientLight (vec3 normal)
{
	vec3 ambientLightDirection = Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);

	if (_IndirectShadingType == 2) // Flatten
	{
		ambientLightDirection = any(_LightColor0) 
		? normalize(_WorldSpaceLightPos0) 
		: ambientLightDirection;
	}

	float ambientLight = dot(normal, ambientLightDirection);
	ambientLight = ambientLight * 0.5 + 0.5;

	if (_IndirectShadingType == 0) // Dynamic
		ambientLight = getGreyscaleSH(normal);
	return ambientLight;
}

SCSS_LightParam initialiseLightParam (SCSS_Light l, vec3 normal, vec3 posWorld)
{
	SCSS_LightParam d = (SCSS_LightParam) 0;
	d.viewDir = normalize(_WorldSpaceCameraPos.xyz - posWorld.xyz);
	d.halfDir = Unity_SafeNormalize (l.dir + d.viewDir);
	d.reflDir = reflect(-d.viewDir, normal); // Calculate reflection vector
	d.NdotL = (dot(l.dir, normal)); // Calculate NdotL
	d.NdotV = (dot(d.viewDir,  normal)); // Calculate NdotV
	d.LdotH = (dot(l.dir, d.halfDir));
	d.NdotH = (dot(normal, d.halfDir)); // Saturate seems to cause artifacts
	d.rlPow4 = Pow4(vec2(dot(d.reflDir, l.dir), 1 - d.NdotV));  
	d.NdotAmb = getAmbientLight(normal);
	return d;
}
#endif
*/

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
	SCSS_RimLightInput rim = (SCSS_RimLightInput) 0;
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

vec2 AnimateTexcoords(vec2 texcoord)
{
	vec2 spriteUV = texcoord;
	if (_UseAnimation)
	{
		_FrameNumber += fract((TIME/20.0) * _AnimationSpeed) * _TotalFrames;

		float frame = clamp(_FrameNumber, 0, _TotalFrames);

		vec2 offPerFrame = vec2((1 / (float)_Columns), (1 / (float)_Rows));

		vec2 spriteSize = texcoord * offPerFrame;

		vec2 currentSprite = 
				vec2(frame * offPerFrame.x,  1 - offPerFrame.y);
		
		float rowIndex;
		float mod = modf(frame / (float)_Columns, rowIndex);
		currentSprite.y -= rowIndex * offPerFrame.y;
		currentSprite.x -= rowIndex * _Columns * offPerFrame.x;
		
		spriteUV = (spriteSize + currentSprite); 
	}
	return spriteUV;

}

vec4 TexCoords(VertexOutput v)
{
    vec4 texcoord;
	texcoord.xy = TRANSFORM_TEX(v.uv0, _MainTex_ST);// Always source from uv0
	texcoord.xy = _PixelSampleMode? 
		sharpSample(_MainTex_TexelSize * _MainTex_ST.xyxy, texcoord.xy) : texcoord.xy;

#if _DETAIL 
	texcoord.zw = TRANSFORM_TEX(((_UVSec == 0) ? v.uv0 : v.uv1), _DetailAlbedoMap_ST);
	texcoord.zw = _PixelSampleMode? 
		sharpSample(_DetailAlbedoMap_TexelSize * _DetailAlbedoMap_ST.xyxy, texcoord.zw) : texcoord.zw;
#else
	texcoord.zw = texcoord.xy;
#endif
    return texcoord;
}

//#define UNITY_SAMPLE_TEX2D_SAMPLER_LOD(tex,samplertex,coord,lod) tex.Sample (sampler##samplertex,coord,lod)
//#define UNITY_SAMPLE_TEX2D_LOD(tex,coord,lod) tex.Sample (sampler##tex,coord,lod)

float OutlineMask(vec2 uv)
{
	// Needs LOD, sampled in vertex function
    return tex2Dlod(_OutlineMask, vec4(uv, 0, 0)).r;
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
	if (_SUBSURFACE == 1) {
		return pow(
			texture (_ThicknessMap, uv).rgb, 
			_ThicknessMapPower);
	} else {
		return 1;
	}
}

vec3 Albedo(vec4 texcoords)
{
    vec3 albedo = texture (_MainTex, texcoords.xy).rgb * LerpWhiteTo(_Color.rgb, ColorMask(texcoords.xy));
#if _DETAIL
    float mask = DetailMask(texcoords.xy);
    vec4 detailAlbedo = texture (_DetailAlbedoMap, texcoords.zw);
    mask *= detailAlbedo.a;
    mask *= _DetailAlbedoMapScale;
    #if _DETAIL_MULX2
        albedo *= LerpWhiteTo (detailAlbedo.rgb * unity_ColorSpaceDouble.rgb, mask);
    #elif _DETAIL_MUL
        albedo *= LerpWhiteTo (detailAlbedo.rgb, mask);
    #elif _DETAIL_ADD
        albedo += detailAlbedo.rgb * mask;
    #elif _DETAIL_LERP
        albedo = lerp (albedo, detailAlbedo.rgb, mask);
    #endif
#endif
    return albedo;
}

float Alpha(vec2 uv)
{
	float alpha = _Color.a;
	switch(_AlbedoAlphaMode)
	{
		case 0: alpha *= texture(_MainTex, uv).a; break;
		case 2: alpha *= texture(_ClippingMask, uv); break;
	}
	return alpha;
}


vec4 SpecularGloss(vec4 texcoords, float mask)
{
    vec4 sg;
	if (_SPECULAR == 1) {
		sg = texture(_SpecGlossMap, texcoords.xy);

		sg.a = _AlbedoAlphaMode == 1? texture(_MainTex, texcoords.xy).a : sg.a;

		sg.rgb *= _SpecColor;
		sg.a *= _Smoothness; // _GlossMapScale is what Standard uses for this
	} else {
		sg = _SpecColor;
		sg.a = _AlbedoAlphaMode == 1? texture(_MainTex, texcoords.xy).a : sg.a;
	}

	if (_DETAIL == 1) {
		vec4 sdm = texture(_SpecularDetailMask,texcoords.zw);
		sg *= saturate4(sdm + 1-(_SpecularDetailStrength*mask));		
	}

    return sg;
}

vec3 Emission(vec2 uv)
{
    return texture(_EmissionMap, uv).rgb;
}

vec4 EmissionDetail(vec2 uv)
{
	if (_DETAIL == 1) {
		uv += _EmissionDetailParams.xy * _Time.y;
		vec4 ed = UNITY_SAMPLE_TEX2D_SAMPLER(_DetailEmissionMap, _DetailAlbedoMap, uv);
		ed.rgb = 
		_EmissionDetailParams.z
		? (sin(ed.rgb * _EmissionDetailParams.w + _Time.y * _EmissionDetailParams.z))+1 
		: ed.rgb;
		return ed;
	} else {
		return 1;
	}
}

vec3 NormalInTangentSpace(vec4 texcoords, float mask)
{
	vec3 normalTangent = UnpackScaleNormal(texture(_BumpMap, TRANSFORM_TEX(texcoords.xy, _MainTex_ST)), _BumpScale);
	if (_DETAIL == 1) {
    	vec3 detailNormalTangent = UnpackScaleNormal(texture (_DetailNormalMap, texcoords.zw), _DetailNormalMapScale);
		if (_DETAIL_LERP == 1) {
			normalTangent = lerp(
				normalTangent,
				detailNormalTangent,
				mask);
		}else {
			normalTangent = lerp(
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
  	const float C = 0;
  	const float D = 1.59;
  	const float E = 0.451;
	color = max((0.0), color - (0.004));
	color = (color * (A * color + B)) / (color * (C * color + D) + E);
	return color;
}

//#if !defined(SCSS_CROSSTONE)
SCSS_TonemapInput Tonemap(vec2 uv, inout float occlusion)
{
	SCSS_TonemapInput t = (SCSS_TonemapInput)0;
	vec4 _ShadowMask_var = texture(_ShadowMask, uv.xy);

	// Occlusion
	if (_ShadowMaskType == 0) 
	{
		// RGB will boost shadow range. Raising _Shadow reduces its influence.
		// Alpha will boost light range. Raising _Shadow reduces its influence.
		t.col = saturate(_IndirectLightingBoost+1.0-_ShadowMask_var.a) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.r;
	}
	// Tone
	if (_ShadowMaskType == 1) 
	{
		t.col = saturate3(_ShadowMask_var.rgb+_IndirectLightingBoost) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.a;
	}
	// Auto-Tone
	if (_ShadowMaskType == 2) 
	{
		vec3 albedo = Albedo(uv.xyxy);
		t.col = saturate3(AutoToneMapping(albedo)+_IndirectLightingBoost) * _ShadowMaskColor.rgb;
		t.bias = _ShadowMaskColor.a*_ShadowMask_var.r;
	}
	t.bias = (1 - _Shadow) * t.bias + _Shadow;
	occlusion = t.bias;
	return t;
}

// Sample ramp with the specified options.
// rampPosition: 0-1 position on the light ramp from light to dark
// softness: 0-1 position on the light ramp on the other axis
vec3 sampleRampWithOptions(float rampPosition, float softness) 
{
	if (_LightRampType == 3) // No sampling
	{
		return saturate(rampPosition*2-1);
	}
	if (_LightRampType == 2) // None
	{
		float shadeWidth = 0.0002 * (1+softness*100);

		const float shadeOffset = 0.5; 
		float lightContribution = simpleSharpen(rampPosition, shadeWidth, shadeOffset);
		return saturate(lightContribution);
	}
	if (_LightRampType == 1) // Vertical
	{
		vec2 rampUV = vec2(softness, rampPosition);
		return tex2D(_Ramp, saturate2(rampUV));
	}
	else // Horizontal
	{
		vec2 rampUV = vec2(rampPosition, softness);
		return tex2D(_Ramp, saturate2(rampUV));
	}
}
//#endif

//#if defined(SCSS_CROSSTONE)
// Tonemaps contain tone in RGB, occlusion in A.
// Midpoint/width is handled in the application function.
SCSS_TonemapInput Tonemap1st (vec2 uv)
{
	vec4 tonemap = texture(_1st_ShadeMap, uv.xy);
	tonemap.rgb = tonemap * _1st_ShadeColor;
	SCSS_TonemapInput t = (SCSS_TonemapInput)1;
	t.col = tonemap.rgb;
	t.bias = tonemap.a;
	return t;
}
SCSS_TonemapInput Tonemap2nd (vec2 uv)
{
	vec4 tonemap = texture(_2nd_ShadeMap, uv.xy);
	tonemap.rgb *= _2nd_ShadeColor;
	SCSS_TonemapInput t = (SCSS_TonemapInput)1;
	t.col = tonemap.rgb;
	t.bias = tonemap.a;
	return t;
}

float adjustShadeMap(float x, float y)
{
	// Might be changed later.
	return (x * (1+y));

}

float ShadingGradeMap (vec2 uv)
{
	vec4 tonemap = texture(_ShadingGradeMap, uv.xy);
	return adjustShadeMap(tonemap.g, _Tweak_ShadingGradeMapLevel);
}
//#endif

float innerOutline (VertexOutput i)
{
	// The compiler should merge this with the later calls.
	// Use the vertex normals for this to avoid artifacts.
	SCSS_LightParam d = initialiseLightParam((SCSS_Light)0, i.normalDir, i.posWorld.xyz);
	float baseRim = d.NdotV;
	baseRim = simpleSharpen(baseRim, 0, _InteriorOutlineWidth * OutlineMask(i.uv0.xy));
	return baseRim;
}

vec3 applyOutline(vec3 col, float is_outline)
{    
	col = lerp(col, col * _outline_color.rgb, is_outline);
    if (_OutlineMode == 2) 
    {
        col = lerp(col, _outline_color.rgb, is_outline);
    }
    return col;
}

SCSS_Input applyOutline(SCSS_Input c, float is_outline)
{

	c.albedo = applyOutline(c.albedo, is_outline);
    if (_CrosstoneToneSeparation == 1) 
    {
        c.tone[0].col = applyOutline(c.tone[0].col, is_outline);
        c.tone[1].col = applyOutline(c.tone[1].col, is_outline);
    }

    return c;
}

void applyVanishing (inout float alpha) {
    const vec3 baseWorldPos = unity_ObjectToWorld._m03_m13_m23;
    float closeDist = distance(_WorldSpaceCameraPos, baseWorldPos);
    float vanishing = saturate(lerpstep(_VanishingStart, _VanishingEnd, closeDist));
    alpha = lerp(alpha, alpha * vanishing, _UseVanishing);
}
