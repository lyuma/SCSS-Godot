shader_type spatial;

// Perform full-quality light calculations on unimportant lights.
// Considering our target GPUs, this is a big visual improvement
// for a small performance penalty.
const SCSS_UNIMPORTANT_LIGHTS_FRAGMENT = 1

// When rendered by a non-HDR camera, clamp incoming lighting.
// This works around issues where scenes are set up incorrectly
// for non-HDR.
const SCSS_CLAMP_IN_NON_HDR = 1

// When screen-space shadows are used in the scene, performs a
// search to find the best sampling point for the shadow
// using the camera's depth buffer. This filters away many aliasing
// artifacts caused by limitations in the screen shadow technique
// used by directional lights.
const SCSS_SCREEN_SHADOW_FILTER = 1

//import "SCSS_UnityGI.shader";
import "SCSS_Utils.shader";
import "SCSS_Input.shader";

// Shade4PointLights from UnityCG.cginc but only returns their attenuation.
// vec4 Shade4PointLightsAtten (
//     vec4 lightPosX, vec4 lightPosY, vec4 lightPosZ,
//     vec4 lightAttenSq,
//     vec3 pos, vec3 normal)
// {
//     // to light vectors
//     vec4 toLightX = lightPosX - pos.x;
//     vec4 toLightY = lightPosY - pos.y;
//     vec4 toLightZ = lightPosZ - pos.z;
//     // squared lengths
//     vec4 lengthSq = 0;
//     lengthSq += toLightX * toLightX;
//     lengthSq += toLightY * toLightY;
//     lengthSq += toLightZ * toLightZ;
//     // don't produce NaNs if some vertex position overlaps with the light
//     lengthSq = max(lengthSq, 0.000001);

//     // NdotL
//     vec4 ndotl = 0;
//     ndotl += toLightX * normal.x;
//     ndotl += toLightY * normal.y;
//     ndotl += toLightZ * normal.z;
//     // correct NdotL
//     vec4 corr = 0;//rsqrt(lengthSq);
//     corr.x = fastRcpSqrtNR0(lengthSq.x);
//     corr.y = fastRcpSqrtNR0(lengthSq.y);
//     corr.z = fastRcpSqrtNR0(lengthSq.z);
//     corr.w = fastRcpSqrtNR0(lengthSq.x);

//     ndotl = corr * (ndotl * 0.5 + 0.5); // Match with Forward for light ramp sampling
//     ndotl = max (vec4(0,0,0,0), ndotl);
//     // attenuation
//     // Fixes popin. Thanks, d4rkplayer!
//     vec4 atten = 1.0 / (1.0 + lengthSq * lightAttenSq);
// 	vec4 atten2 = saturate(1 - (lengthSq * lightAttenSq / 25));
// 	atten = min(atten, atten2 * atten2);

//     vec4 diff = ndotl * atten;
//     if (SCSS_UNIMPORTANT_LIGHTS_FRAGMENT) {
//     return atten;
//     #else
//     return diff;
//     }
// }

// Based on Standard Shader's forwardbase vertex lighting calculations in VertexGIForward
// This revision does not pass the light values themselves, but only their attenuation.
// vec4 VertexLightContribution(vec3 posWorld, vec3 normalWorld)
// {
// 	vec4 vertexLight = 0;

// 	// Static lightmapped materials are not allowed to have vertex lights.
// 	#ifdef LIGHTMAP_ON
// 		return 0;
// 	#elif UNITY_SHOULD_SAMPLE_SH
// 		#ifdef VERTEXLIGHT_ON
// 			// Approximated illumination from non-important point lights
// 			vertexLight = Shade4PointLightsAtten(
// 				unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
// 				unity_4LightAtten0, posWorld, normalWorld);
// 		}
// 	}

// 	return vertexLight;
// }

//SSS method from GDC 2011 conference by Colin Barre-Bresebois & Marc Bouchard and modified by Xiexe
vec3 getSubsurfaceScatteringLight (SCSS_Light l, vec3 normalDirection, vec3 viewDirection, 
    float attenuation, vec3 thickness, vec3 indirectLight = 0)
{
    vec3 vSSLight = l.dir + normalDirection * _SSSDist; // Distortion
    vec3 vdotSS = pow(saturate(dot(viewDirection, -vSSLight)), _SSSPow) 
        * _SSSIntensity; 
    
    return lerp(1, attenuation, float(any(_WorldSpaceLightPos0.xyz))) 
                * (vdotSS + _SSSAmbient) * abs(_ThicknessMapInvert-thickness)
                * (l.color + indirectLight) * _SSSCol;
                
}

vec3 sampleCrossToneLighting(inout float x, SCSS_TonemapInput tone0, SCSS_TonemapInput tone1, vec3 albedo)
{
	// A three-tiered tone system.
	// Input x is potentially affected by occlusion map.

	x = x;
	float offset0 = _1st_ShadeColor_Step * tone0.bias; 
	float width0  = _1st_ShadeColor_Feather;
	float factor0 = saturate(simpleSharpen(x, width0, offset0));

	float offset1 = _2nd_ShadeColor_Step * tone1.bias; 
	float width1  = _2nd_ShadeColor_Feather;
	float factor1 = saturate(simpleSharpen(x, width1, offset1));

	vec3 final;
	final = lerp(tone1.col, tone0.col, factor1);

	if (_CrosstoneToneSeparation == 0) 	final = lerp(final, 1.0, factor0) * albedo;
	if (_CrosstoneToneSeparation == 1) 	final = lerp(final, albedo, factor0);
	
	x = factor0;
	
	return final;
}

float applyShadowLift(float baseLight, float occlusion)
{
	baseLight *= occlusion;
	baseLight = _ShadowLift + baseLight * (1-_ShadowLift);
	return baseLight;
}

float applyShadowLift(vec4 baseLight, float occlusion)
{
	baseLight *= occlusion;
	baseLight = _ShadowLift + baseLight * (1-_ShadowLift);
	return baseLight;
}

float getRemappedLight(float perceptualRoughness, SCSS_LightParam d)
{
	float diffuseShadowing = DisneyDiffuse(abs(d.NdotV), abs(d.NdotL), d.LdotH, perceptualRoughness);
	float remappedLight = d.NdotL * LerpOneTo(diffuseShadowing, _DiffuseGeomShadowFactor);
	return remappedLight;
}

float applyAttenuation(float NdotL, float attenuation)
{
	if (SCSS_CROSSTONE) {
	//attenuation = round(attenuation);
	float shadeVal = _1st_ShadeColor_Step - _1st_ShadeColor_Feather * 0.5;
	shadeVal = shadeVal-0.01;
	//NdotL = min(lerp(shadeVal, NdotL, attenuation), NdotL);
	NdotL = lerp(shadeVal*NdotL, NdotL, attenuation);
	#else
	NdotL = min(NdotL * attenuation, NdotL);
	//NdotL = lerp(0.5, NdotL, attenuation);
	}
	return NdotL;
}

// vec3 calcVertexLight(vec4 vertexAttenuation, float occlusion, SCSS_TonemapInput tone[2], float softness)
// {
// 	vec3 vertexContribution = 0;
// 	if (UNITY_PASS_FORWARDBASE) {

// 		if (!SCSS_CROSSTONE) {
// 		vertexAttenuation = applyShadowLift(vertexAttenuation, occlusion);
//     	for (int num = 0; num < 4; num++) {
//     		vertexContribution += unity_LightColor[num] * 
//     			(sampleRampWithOptions(vertexAttenuation[num], softness)+tone[0].col);
//     	}
//     	}

// 		if (SCSS_CROSSTONE) {
//     	for (int num = 0; num < 4; num++) {
//     		vertexContribution += unity_LightColor[num] * 
//     			sampleCrossToneLighting(vertexAttenuation[num], tone[0], tone[1], 1.0);
//     	}
//     	}

// 	}
// 	return vertexContribution;
// }

void getDirectIndirectLighting(vec3 normal, out vec3 directLighting, out vec3 indirectLighting)
{
	directLighting = 0.0;
	indirectLighting = 0.0;
	switch (_LightingCalculationType)
	{
	case 0: // Arktoon
		directLighting   = GetSHLength();
		indirectLighting = BetterSH9(vec4(0.0, 0.0, 0.0, 1.0)); 
	break;
	case 1: // Standard
		directLighting = 
		indirectLighting = BetterSH9(vec4(normal, 1.0));
	break;
	case 2: // Cubed
		directLighting   = BetterSH9(vec4(0.0,  1.0, 0.0, 1.0));
		indirectLighting = BetterSH9(vec4(0.0, -1.0, 0.0, 1.0)); 
	break;
	case 3: // True Directional
		vec4 ambientDir = vec4(Unity_SafeNormalize(unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz), 1.0);
		directLighting   = BetterSH9(ambientDir);
		indirectLighting = BetterSH9(-ambientDir); 
	break;
	}
}

// For baked lighting.
vec3 calcDiffuseGI(vec3 albedo, SCSS_TonemapInput tone[2], float occlusion, float softness,
	vec3 indirectLighting, vec3 directLighting, SCSS_LightParam d)
{
	float ambientLight = d.NdotAmb;
	
	vec3 indirectAverage = 0.5 * (indirectLighting + directLighting);

	// Make this a UI value later.
	const float ambientLightSplitThreshold = 1.0/1.0;
	float ambientLightSplitFactor = 
	saturate(
		dot(abs((directLighting-indirectLighting)/indirectAverage), 
		ambientLightSplitThreshold * sRGB_Luminance));

	if (!SCSS_CROSSTONE) {
	ambientLight = applyShadowLift(ambientLight, occlusion);
	vec3 indirectContribution = sampleRampWithOptions(ambientLight, softness);
	indirectLighting = lerp(indirectLighting, directLighting, tone[0].col);
	indirectAverage = lerp(indirectAverage, directLighting, tone[0].col);
	}

	if (SCSS_CROSSTONE) {
	ambientLight *= occlusion;
	indirectAverage *= albedo;
	vec3 indirectContribution = sampleCrossToneLighting(ambientLight, tone[0], tone[1], albedo);
	}

	vec3 lightContribution;

	if (SCSS_CROSSTONE) {
	if (_CrosstoneToneSeparation == 0) lightContribution = 
	lerp(indirectAverage,
	lerp(indirectLighting, directLighting, indirectContribution),
	ambientLightSplitFactor) * albedo;

	if (_CrosstoneToneSeparation == 1) lightContribution = 
	lerp(indirectAverage,
	directLighting*indirectContribution,
	ambientLightSplitFactor);
	}

	if (!SCSS_CROSSTONE) {
	lightContribution = 
	lerp(indirectAverage,
	lerp(indirectLighting, directLighting, indirectContribution),
	ambientLightSplitFactor) * albedo;
	}

	return lightContribution;
}

// For directional lights where attenuation is shadow.
vec3 calcDiffuseBase(vec3 albedo, SCSS_TonemapInput tone[2], float occlusion, float perceptualRoughness, 
	float attenuation, float softness, SCSS_LightParam d, SCSS_Light l)
{
	float remappedLight = getRemappedLight(perceptualRoughness, d);
	remappedLight = remappedLight * 0.5 + 0.5;

	remappedLight = applyAttenuation(remappedLight, attenuation);

	if (!SCSS_CROSSTONE) {
	remappedLight = applyShadowLift(remappedLight, occlusion);
	vec3 lightContribution = lerp(tone[0].col, 1.0, sampleRampWithOptions(remappedLight, softness)) * albedo;
	}

	if (SCSS_CROSSTONE) {
	remappedLight *= occlusion;
	vec3 lightContribution = sampleCrossToneLighting(remappedLight, tone[0], tone[1], albedo);
	}

	lightContribution *= l.color;

	return lightContribution;	
}

// For point/spot lights where attenuation is shadow+attenuation.
vec3 calcDiffuseAdd(vec3 albedo, SCSS_TonemapInput tone[2], float occlusion, float perceptualRoughness, 
	float softness, SCSS_LightParam d, SCSS_Light l)
{
	float remappedLight = getRemappedLight(perceptualRoughness, d);
	remappedLight = remappedLight * 0.5 + 0.5;

	if (!SCSS_CROSSTONE) {
	remappedLight = applyShadowLift(remappedLight, occlusion);
	vec3 lightContribution = sampleRampWithOptions(remappedLight, softness);

	vec3 directLighting = l.color;
	vec3 indirectLighting = l.color * tone[0].col;

	lightContribution = lerp(indirectLighting, directLighting, lightContribution) * albedo;
	}

	if (SCSS_CROSSTONE) {
	vec3 lightContribution = sampleCrossToneLighting(remappedLight, tone[0], tone[1], albedo);
	lightContribution *= l.color;
	}

	return lightContribution;
}

void getSpecularVD(float roughness, SCSS_LightParam d, SCSS_Light l, VertexOutput i,
	out float V, out float D)
{
	V = 0; D = 0;

	#ifndef SHADER_TARGET_GLSL
	[call]
	}
	switch(_SpecularType)
	{
	case 1: // GGX
		V = SmithJointGGXVisibilityTerm (d.NdotL, d.NdotV, roughness);
	    D = GGXTerm (d.NdotH, roughness);
	    break;

	case 2: // Charlie (cloth)
		V = V_Neubelt (d.NdotV, d.NdotL);
	    D = D_Charlie (roughness, d.NdotH);
	    break;

	case 3: // GGX anisotropic
	    float anisotropy = _Anisotropy;
	    float at = max(roughness * (1.0 + anisotropy), 0.002);
	    float ab = max(roughness * (1.0 - anisotropy), 0.002);

		#if 0
	    float TdotL = dot(i.tangentDir, l.dir);
	    float BdotL = dot(i.bitangentDir, l.dir);
	    float TdotV = dot(i.tangentDir, viewDir);
	    float BdotV = dot(i.bitangentDir, l.dir);

	    // Accurate but probably expensive
		V = V_SmithGGXCorrelated_Anisotropic (at, ab, TdotV, BdotV, TdotL, BdotL, d.NdotV, d.NdotL);
		#else
		V = SmithJointGGXVisibilityTerm (d.NdotL, d.NdotV, roughness);
		}
		// Temporary
	    D = D_GGX_Anisotropic(d.NdotH, d.halfDir, i.tangentDir, i.bitangentDir, at, ab);
	    break;
	}
	return;
}

vec3 calcSpecularBase(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	float attenuation, float occlusion, SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	UnityGI gi = (UnityGI)0;
	
	float V = 0; float D = 0; 
	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	// "GGX with roughness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughness remapping."
	// This also fixes issues with the other specular types.
	roughness = max(roughness, 0.002);

	d = saturate(d);

	getSpecularVD(roughness, d, l, i, /*out*/ V, /*out*/ D);

	float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later
	specularTerm = max(0, specularTerm * d.NdotL);

	if (_SPECULARHIGHLIGHTS_OFF) {
    	specularTerm = 0.0;
	}

	float surfaceReduction = 1.0 / (roughness*roughness + 1);

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(specColor) ? 1.0 : 0.0;

	gi =  GetUnityGI(l.color.rgb, l.dir, normal, 
		d.viewDir, d.reflDir, attenuation, occlusion, perceptualRoughness, i.posWorld.xyz);

	float grazingTerm = saturate(smoothness + (1-oneMinusReflectivity));

	return
	specularTerm * (gi.light.color) * FresnelTerm(specColor, d.LdotH) +
	surfaceReduction * (gi.indirect.specular.rgb) * FresnelLerp(specColor, grazingTerm, d.NdotV);
	
}

vec3 calcSpecularBase(SCSS_Input c, float perceptualRoughness, float attenuation,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularBase(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, 
		perceptualRoughness, attenuation, c.occlusion, d, l, i);
}

vec3 calcSpecularAdd(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	if (_SPECULARHIGHLIGHTS_OFF) {
		return 0.0;
	}
	
	float V = 0; float D = 0; 
	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	// "GGX with roughness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughness remapping."
	// This also fixes issues with the other specular types.
	roughness = max(roughness, 0.002);

	d = saturate(d);

	getSpecularVD(roughness, d, l, i, /*out*/ V, /*out*/ D);

	float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later
	specularTerm = max(0, specularTerm * d.NdotL);

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(specColor) ? 1.0 : 0.0;

	return
	specularTerm * l.color * FresnelTerm(specColor, d.LdotH);
	
}

vec3 calcSpecularAdd(SCSS_Input c, float perceptualRoughness,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularAdd(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, perceptualRoughness, d, l, i);
}

vec3 calcSpecularCel(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	float attenuation, SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	if (_SpecularType == 4) {
		// 
		float spec = max(d.NdotH, 0);
		spec = pow(spec, (smoothness)*40) * _CelSpecularSteps;
		spec = sharpenLighting(fract(spec), _CelSpecularSoftness)+floor(spec);
    	spec = max(0.02,spec);
    	spec *= UNITY_PI * rcp(_CelSpecularSteps);

    	vec3 envLight = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, normal, UNITY_SPECCUBE_LOD_STEPS);
		return (spec * specColor *  l.color) + (spec * specColor * envLight);
	}
	if (_SpecularType == 5) {
		vec3 strandTangent = (_Anisotropy < 0)
		? i.tangentDir
		: i.bitangentDir;
		_Anisotropy = abs(_Anisotropy);
		strandTangent = lerp(normal, strandTangent, _Anisotropy);
		float exponent = smoothness;
		float spec  = StrandSpecular(strandTangent, d.halfDir, 
			exponent*80, 1.0 );
		float spec2 = StrandSpecular(strandTangent, d.halfDir, 
			exponent*40, 0.5 );
		spec  = sharpenLighting(fract(spec), _CelSpecularSoftness)+floor(spec);
		spec2 = sharpenLighting(fract(spec2), _CelSpecularSoftness)+floor(spec2);
		spec += spec2;
		
    	vec3 envLight = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, normal, UNITY_SPECCUBE_LOD_STEPS);
		return (spec * specColor *  l.color) + (spec * specColor * envLight);
	}
	return 0;
}

vec3 calcSpecularCel(SCSS_Input c, float perceptualRoughness, float attenuation, SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularCel(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, perceptualRoughness, attenuation, d, l, i);
}

vec3 SCSS_ShadeBase(SCSS_Input c, VertexOutput i, SCSS_Light l, float attenuation)
{	
	vec3 finalColor;

	float isOutline = i.extraData.x;

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz);

	vec3 directLighting, indirectLighting;

	getDirectIndirectLighting(c.normal, /*out*/ directLighting, /*out*/ indirectLighting);

	finalColor  = calcDiffuseGI(c.albedo, c.tone, c.occlusion, c.softness, indirectLighting, directLighting, d);
	finalColor += calcDiffuseBase(c.albedo, c.tone, c.occlusion,
		c.perceptualRoughness, attenuation, c.softness, d, l);

	// Prepare fake light params for subsurface scattering.
	SCSS_Light iL = l;
	SCSS_LightParam iD = d;
	iL.color = GetSHLength();
	iL.dir = Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);
	iD = initialiseLightParam(iL, c.normal, i.posWorld.xyz);

	// Prepare fake light params for spec/fresnel which simulate specular.
	SCSS_Light fL = l;
	SCSS_LightParam fD = d;
	fL.color = attenuation * fL.color + GetSHLength();
	fL.dir = Unity_SafeNormalize(fL.dir + (unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);
	fD = initialiseLightParam(fL, c.normal, i.posWorld.xyz);

	if (isOutline <= 0)
	{
		if (_SUBSURFACE) {
			if (USING_DIRECTIONAL_LIGHT) {
			finalColor += getSubsurfaceScatteringLight(l, c.normal, d.viewDir,
				attenuation, c.thickness) * c.albedo;
			}
		finalColor += getSubsurfaceScatteringLight(iL, c.normal, iD.viewDir,
			1, c.thickness) * c.albedo;
		}

		if (_METALLICGLOSSMAP) {
	    finalColor += calcSpecularBase(c, c.perceptualRoughness, attenuation, d, l, i);
	    }

	    if (_SPECGLOSSMAP) {
    	finalColor += calcSpecularCel(c, c.perceptualRoughness, attenuation, fD, fL, i);
   		}
    };

    return finalColor;
}

vec3 SCSS_ShadeLight(SCSS_Input c, VertexOutput i, SCSS_Light l, float attenuation)
{
	vec3 finalColor;

	float isOutline = i.extraData.x;

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz);

    finalColor = calcDiffuseAdd(c.albedo, c.tone, c.occlusion, c.perceptualRoughness, c.softness, d, l);

	if (isOutline <= 0)
	{
		if (_SUBSURFACE) { 
		finalColor += c.albedo * getSubsurfaceScatteringLight(l, c.normal, d.viewDir,
			attenuation, c.thickness, c.tone[0].col);
		}

		if (_METALLICGLOSSMAP) {
    	finalColor += calcSpecularAdd(c, c.perceptualRoughness, d, l, i);
    	}

		if (_SPECGLOSSMAP) {
		finalColor += calcSpecularCel(c, c.perceptualRoughness, attenuation, d, l, i);
		}
	};
	return finalColor;
}


vec3 SCSS_ApplyLighting(SCSS_Input c, VertexOutput i, vec4 texcoords, float attenuation)
{
	UNITY_LIGHT_ATTENUATION(attenuation, i, i.posWorld.xyz);

	// FIXME.
	//if (SCSS_SCREEN_SHADOW_FILTER) && defined(USING_SHADOWS_UNITY) {
	//correctedScreenShadowsForMSAA(i._ShadowCoord, attenuation);
	//}


	float isOutline = i.extraData.x;

	// Lighting parameters
	SCSS_Light l = MainLight(i.posWorld.xyz);
	if (UNITY_PASS_FORWARDADD) && !defined(USING_DIRECTIONAL_LIGHT) {
	l.dir = normalize(_WorldSpaceLightPos0.xyz - i.posWorld.xyz);
	} 

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz);

	if (_METALLICGLOSSMAP) {
	// Geometric Specular AA from HDRP
	c.smoothness = GeometricNormalFiltering(c.smoothness, i.normalDir.xyz, 0.25, 0.5);
	}

	if (_METALLICGLOSSMAP) {
	// Perceptual roughness transformation. Without this, roughness handling is wrong.
	c.perceptualRoughness = SmoothnessToPerceptualRoughness(c.smoothness);
	} else {
	// Disable DisneyDiffuse for cel specular.
	}

	// Generic lighting for matcaps/rimlighting. 
	// Currently matcaps are applied to albedo, so they don't need lighting. 
	vec3 effectLighting = l.color;
	if (UNITY_PASS_FORWARDBASE) {
	//effectLighting *= attenuation;
	effectLighting += BetterSH9(vec4(0.0,  0.0, 0.0, 1.0));
	}

	// Apply matcap before specular effect.
	if (_UseMatcap >= 1 && isOutline <= 0) 
	{
		vec2 matcapUV;
		if (_UseMatcap == 1) matcapUV = getMatcapUVsOriented(c.normal, d.viewDir, vec3(0, 1, 0));
		if (_UseMatcap == 2) matcapUV = getMatcapUVsOriented(c.normal, d.viewDir, i.bitangentDir.xyz);

		vec4 _MatcapMask_var = MatcapMask(i.uv0.xy);
		c.albedo = applyMatcap(_Matcap1, matcapUV, c.albedo, 1.0, _Matcap1Blend, _Matcap1Strength * _MatcapMask_var.r);
		c.albedo = applyMatcap(_Matcap2, matcapUV, c.albedo, 1.0, _Matcap2Blend, _Matcap2Strength * _MatcapMask_var.g);
		c.albedo = applyMatcap(_Matcap3, matcapUV, c.albedo, 1.0, _Matcap3Blend, _Matcap3Strength * _MatcapMask_var.b);
		c.albedo = applyMatcap(_Matcap4, matcapUV, c.albedo, 1.0, _Matcap4Blend, _Matcap4Strength * _MatcapMask_var.a);
	}

	vec3 finalColor = 0; 

	float fresnelLightMaskBase = LerpOneTo((d.NdotH), _UseFresnelLightMask);
	float fresnelLightMask = 
		saturate(pow(saturate( fresnelLightMaskBase), _FresnelLightMask));
	float fresnelLightMaskInv = 
		saturate(pow(saturate(-fresnelLightMaskBase), _FresnelLightMask));
	
	// Lit
	if (_UseFresnel == 1 && isOutline <= 0) 
	{
		vec3 sharpFresnel = sharpenLighting(d.rlPow4.y * c.rim.width * fresnelLightMask, 
			c.rim.power) * c.rim.tint * c.rim.alpha;
		sharpFresnel += sharpenLighting(d.rlPow4.y * c.rim.invWidth * fresnelLightMaskInv,
			c.rim.invPower) * c.rim.invTint * c.rim.invAlpha * _FresnelLightMask;
		c.albedo += c.albedo * sharpFresnel;
	}

	// AmbientAlt
	if (_UseFresnel == 3 && isOutline <= 0)
	{
		float sharpFresnel = sharpenLighting(d.rlPow4.y*c.rim.width, c.rim.power)
		 * c.rim.alpha;
		c.occlusion += saturate(sharpFresnel);
	}

	if (_UseFresnel == 4 && isOutline <= 0)
	{
		vec3 sharpFresnel = sharpenLighting(d.rlPow4.y * c.rim.width * fresnelLightMask, 
			c.rim.power);
		c.occlusion += saturate3(sharpFresnel);
		sharpFresnel *= c.rim.tint * c.rim.alpha;
		sharpFresnel += sharpenLighting(d.rlPow4.y * c.rim.invWidth * fresnelLightMaskInv,
			c.rim.invPower) * c.rim.invTint * c.rim.invAlpha * _FresnelLightMask;
		c.albedo += c.albedo * sharpFresnel;
	}

	if (UNITY_PASS_FORWARDBASE) {
	finalColor = SCSS_ShadeBase(c, i, l, attenuation);
	}

	if (UNITY_PASS_FORWARDADD) {
	finalColor = SCSS_ShadeLight(c, i, l, attenuation);
	}

	// Proper cheap vertex lights. 
	//if (VERTEXLIGHT_ON) && !defined(SCSS_UNIMPORTANT_LIGHTS_FRAGMENT) {
	//finalColor += c.albedo * calcVertexLight(i.vertexLight, c.occlusion, c.tone, c.softness);
	//}

	// Ambient
	if (_UseFresnel == 2 && isOutline <= 0)
	{
		vec3 sharpFresnel = sharpenLighting(d.rlPow4.y * c.rim.width * fresnelLightMask, 
			c.rim.power) * c.rim.tint * c.rim.alpha;
		sharpFresnel += sharpenLighting(d.rlPow4.y * c.rim.invWidth * fresnelLightMaskInv,
			c.rim.invPower) * c.rim.invTint * c.rim.invAlpha * _FresnelLightMask;
		finalColor += effectLighting*sharpFresnel;
	}

	//vec3 wrappedDiffuse = LightColour * saturate((dot(N, L) + w) / ((1 + w) * (1 + w)));

    //// Workaround for scenes with HDR off blowing out in VRchat.
    //#if !UNITY_HDR_ON && SCSS_CLAMP_IN_NON_HDR
    //    l.color = saturate3(l.color);
    //}

    // Apply full lighting to unimportant lights. This is cheaper than you might expect.
	if (UNITY_PASS_FORWARDBASE) && defined(VERTEXLIGHT_ON) && defined(SCSS_UNIMPORTANT_LIGHTS_FRAGMENT) {
    for (int num = 0; num < 4; num++) {
    	UNITY_BRANCH if ((unity_LightColor[num].r + unity_LightColor[num].g + unity_LightColor[num].b + i.vertexLight[num]) != 0.0)
    	{
    	l.color = unity_LightColor[num].rgb;
    	l.dir = normalize(vec3(unity_4LightPosX0[num], unity_4LightPosY0[num], unity_4LightPosZ0[num]) - i.posWorld.xyz);

		finalColor += SCSS_ShadeLight(c, i, l, 1) *  i.vertexLight[num];	
    	}
    };
	}

	if (UNITY_PASS_FORWARDADD) {
		finalColor *= attenuation;
	}

	finalColor *= _LightWrappingCompensationFactor;

	if (UNITY_PASS_FORWARDBASE) {
	vec3 emission;
	vec4 emissionDetail = EmissionDetail(texcoords.zw);

	finalColor = max(0, finalColor - saturate((1-emissionDetail.w)- (1-c.emission)));
	emission = emissionDetail.rgb * c.emission * _EmissionColor.rgb;

	// Emissive c.rim. To restore masking behaviour, multiply by emissionMask.
	emission += _CustomFresnelColor.xyz * (pow(d.rlPow4.y, rcp(_CustomFresnelColor.w+0.0001)));

	emission *= (1-isOutline);
	finalColor += emission;
	}

	return finalColor;
}
