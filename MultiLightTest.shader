shader_type spatial;
render_mode cull_disabled;
//import "res://SCSS/Shaders/SCSS_Core.shader";

// Perform full-quality light calculations on unimportant lights.
// Considering our target GPUs, this is a big visual improvement
// for a small performance penalty.
//const SCSS_UNIMPORTANT_LIGHTS_FRAGMENT = 1

// When rendered by a non-HDR camera, clamp incoming lighting.
// This works around issues where scenes are set up incorrectly
// for non-HDR.
//const SCSS_CLAMP_IN_NON_HDR = 1

// When screen-space shadows are used in the scene, performs a
// search to find the best sampling point for the shadow
// using the camera's depth buffer. This filters away many aliasing
// artifacts caused by limitations in the screen shadow technique
// used by directional lights.
//const SCSS_SCREEN_SHADOW_FILTER = 1

//import "SCSS_UnityGI.shader";


// Portability functions. Overloads not possible.
vec4 saturate4(vec4 x) {
    return clamp(x, vec4(0.0), vec4(1.0));
}

vec3 saturate3(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}

vec2 saturate2(vec2 x) {
    return clamp(x, vec2(0.0), vec2(1.0));
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 Unity_SafeNormalize(vec3 x) {
    return all(equal(x, vec3(0.0))) ? x : normalize(x);
}

vec3 Pow4 (vec3 x)
{
    return x*x*x*x;
}

vec2 Pow4_v2 (vec2 x)
{
    return x*x*x*x;
}

float Pow4_f (float x)
{
    return x*x*x*x;
}
vec3 Pow5 (vec3 x)
{
    return x*x * x*x * x;
}

float Pow5_f (float x)
{
    return x*x * x*x * x;
}

vec2 TRANSFORM_TEX(vec2 uv, vec4 st)
{
	return uv * st.xy + st.zw;
}

vec3 FresnelTerm (vec3 F0, float cosA)
{
    float t = Pow5_f (1.0 - cosA);   // ala Schlick interpoliation
    return F0 + (1.0-F0) * t;
}
vec3 FresnelLerp (vec3 F0, vec3 F90, float cosA)
{
    float t = Pow5_f (1.0 - cosA);   // ala Schlick interpoliation
    return mix (F0, F90, t);
}
// approximage Schlick with ^4 instead of ^5
vec3 FresnelLerpFast (vec3 F0, vec3 F90, float cosA)
{
    float t = Pow4_f (1.0 - cosA);
    return mix (F0, F90, t);
}


const bool _ALPHATEST_ON = true;

const vec3 sRGB_Luminance = vec3(0.2126, 0.7152, 0.0722);
const float UNITY_PI = 3.14159265358979;
const float UNITY_INV_PI = 1.0 / UNITY_PI;

struct SCSS_Light
{
	vec3 cameraPos; // May be in world space or view space depending on light.
    vec3 color;
    vec3 dir;
    float intensity;
    float attenuation;
    //bool isForwardAdd;
    //bool isDirectional;
};

struct VertexOutput
{
    vec3 posWorld;
    vec3 normalDir;
    vec3 tangentDir;
    vec3 bitangentDir;
    vec4 extraData;
    vec2 uv0;
};


float interleaved_gradient(vec2 uv)
{
	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	return fract(magic.z * fract(dot(uv, magic.xy)));
}

float Dither17(vec2 Pos, float FrameIndexMod4)
{
    // 3 scalar float ALU (1 mul, 2 mad, 1 fract)
    return fract(dot(vec3(Pos.xy, FrameIndexMod4), vec3(uvec3(2, 7, 23)) / 17.0f));
}

float max3 (vec3 x) 
{
	return max(x.x, max(x.y, x.z));
}

// "R2" dithering

// Triangle Wave
float T(float z) {
    return z >= 0.5 ? 2.-2.*z : 2.*z;
}

// R dither mask
float intensity(vec2 pixel) {
    const float a1 = 0.75487766624669276;
    const float a2 = 0.569840290998;
    return fract(a1 * float(pixel.x) + a2 * float(pixel.y));
}

float rDither(float gray, vec2 pos) {
	const float steps = 4.0;
	// pos is screen pixel position in 0-res range
    // Calculated noised gray value
    float noised = (2./float(steps)) * T(intensity(vec2(pos.xy))) + gray - (1./float(steps));
    // Clamp to the number of gray levels we want
    return floor(float(steps) * noised) / (float(steps)-1.);
}

float scene_time = 0.0;

// "R2" dithering -- endtime

bool applyAlphaClip(float time, inout float alpha, float cutoff, vec2 pos, bool sharpen)
{
    pos += vec2(mod(sin(time/8.0),4.0));
    if (_ALPHATEST_ON) { //#if defined(_ALPHATEST_ON)
    // Switch between dithered alpha and sharp-edge alpha.
        if (!sharpen) {
            float mask = (T(intensity(pos)));
            alpha = saturate(alpha + alpha * mask); 
        }
        else {
            alpha = ((alpha - cutoff) / max(fwidth(alpha), 0.0001) + 0.5);
        }
        // If 0, remove now.
        if (alpha < 0.0) {
			return true;
		}
    } //#endif
	return false;
}

vec3 BlendNormalsPD(vec3 n1, vec3 n2) {
	return normalize(vec3(n1.xy*n2.z + n2.xy*n1.z, n1.z*n2.z));
}

vec2 invlerp(vec2 A, vec2 B, vec2 T){
    return (T - A)/(B - A);
}

// Returns pixel sharpened to nearest pixel boundary. 
// texelSize is Unity _Texture_TexelSize; zw is w/h, xy is 1/wh
vec2 sharpSample( vec4 texelSize , vec2 p )
{
	p = p*texelSize.zw;
    vec2 c = max(vec2(0.0001), fwidth(p));
    p = floor(p) + saturate2(fract(p) / c);
	p = (p - 0.5)*texelSize.xy;
	return p;
}

// bool inMirror()
// {
// 	return unity_CameraProjection[2][0] != 0.f || unity_CameraProjection[2][1] != 0.f;
// }

//-----------------------------------------------------------------------------
// Helper functions for roughness
//-----------------------------------------------------------------------------

float PerceptualRoughnessToRoughness(float perceptualRoughness)
{
    return perceptualRoughness * perceptualRoughness;
}

float RoughnessToPerceptualRoughness(float roughness)
{
    return sqrt(roughness);
}

float RoughnessToPerceptualSmoothness(float roughness)
{
    return 1.0 - sqrt(roughness);
}

float PerceptualSmoothnessToRoughness(float perceptualSmoothness)
{
    return (1.0 - perceptualSmoothness) * (1.0 - perceptualSmoothness);
}

float PerceptualSmoothnessToPerceptualRoughness(float perceptualSmoothness)
{
    return (1.0 - perceptualSmoothness);
}

float PerceptualRoughnessToPerceptualSmoothness(float perceptualRoughness)
{
    return (1.0 - perceptualRoughness);
}

// Smoothness is the user facing name
// it should be perceptualSmoothness but we don't want the user to have to deal with this name
float SmoothnessToRoughness(float smoothness)
{
    return (1.0 - smoothness) * (1.0 - smoothness);
}

float SmoothnessToPerceptualRoughness(float smoothness)
{
    return (1.0 - smoothness);
}


// Return modified perceptualSmoothness based on provided variance (get from GeometricNormalVariance + TextureNormalVariance)
float NormalFiltering(float perceptualSmoothness, float variance, float threshold)
{
    float roughness = PerceptualSmoothnessToRoughness(perceptualSmoothness);
    // Ref: Geometry into Shading - http://graphics.pixar.com/library/BumpRoughness/paper.pdf - equation (3)
    float squaredRoughness = saturate(roughness * roughness + min(2.0 * variance, threshold * threshold)); // threshold can be really low, square the value for easier control

    return RoughnessToPerceptualSmoothness(sqrt(squaredRoughness));
}

// Reference: Error Reduction and Simplification for Shading Anti-Aliasing
// Specular antialiasing for geometry-induced normal (and NDF) variations: Tokuyoshi / Kaplanyan et al.'s method.
// This is the deferred approximation, which works reasonably well so we keep it for forward too for now.
// screenSpaceVariance should be at most 0.5^2 = 0.25, as that corresponds to considering
// a gaussian pixel reconstruction kernel with a standard deviation of 0.5 of a pixel, thus 2 sigma covering the whole pixel.
float GeometricNormalVariance(vec3 geometricNormalWS, float screenSpaceVariance)
{
    vec3 deltaU = dFdx(geometricNormalWS);
    vec3 deltaV = dFdy(geometricNormalWS);

    return screenSpaceVariance * (dot(deltaU, deltaU) + dot(deltaV, deltaV));
}

// Return modified perceptualSmoothness
float GeometricNormalFiltering(float perceptualSmoothness, vec3 geometricNormalWS, float screenSpaceVariance, float threshold)
{
    float variance = GeometricNormalVariance(geometricNormalWS, screenSpaceVariance);
    return NormalFiltering(perceptualSmoothness, variance, threshold);
}

// bgolus's method for "fixing" screen space directional shadows and anti-aliasing
// https://forum.unity.com/threads/fixing-screen-space-directional-shadows-and-anti-aliasing.379902/
// Searches the depth buffer for the depth closest to the current fragment to sample the shadow from.
// This reduces the visible aliasing. 
/*
void correctedScreenShadowsForMSAA(vec4 _ShadowCoord, inout float shadow)
{
    #ifdef SHADOWS_SCREEN

    vec2 screenUV = _ShadowCoord.xy / _ShadowCoord.w;
    shadow = tex2D(_ShadowMapTexture, screenUV).r;

    float fragDepth = _ShadowCoord.z / _ShadowCoord.w;
    float depth_raw = tex2D(_CameraDepthTexture, screenUV).r;

    float depthDiff = abs(fragDepth - depth_raw);
    float diffTest = 1.0 / 100000.0;

    if (depthDiff > diffTest)
    {
        vec2 texelSize = _CameraDepthTexture_TexelSize.xy;
        vec4 offsetDepths = 0;

        vec2 uvOffsets[5] = {
            vec2(1.0, 0.0) * texelSize,
            vec2(-1.0, 0.0) * texelSize,
            vec2(0.0, 1.0) * texelSize,
            vec2(0.0, -1.0) * texelSize,
            vec2(0.0, 0.0)
        };

        offsetDepths.x = tex2D(_CameraDepthTexture, screenUV + uvOffsets[0]).r;
        offsetDepths.y = tex2D(_CameraDepthTexture, screenUV + uvOffsets[1]).r;
        offsetDepths.z = tex2D(_CameraDepthTexture, screenUV + uvOffsets[2]).r;
        offsetDepths.w = tex2D(_CameraDepthTexture, screenUV + uvOffsets[3]).r;

        vec4 offsetDiffs = abs(fragDepth - offsetDepths);

        float diffs[4] = {offsetDiffs.x, offsetDiffs.y, offsetDiffs.z, offsetDiffs.w};

        int lowest = 4;
        float tempDiff = depthDiff;
        for (int i=0; i<4; i++)
        {
            if(diffs[i] < tempDiff)
            {
                tempDiff = diffs[i];
                lowest = i;
            }
        }

        shadow = tex2D(_ShadowMapTexture, screenUV + uvOffsets[lowest]).r;
    }
    #endif //SHADOWS_SCREEN
}
*/
// RCP SQRT
// Source: https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h

const int IEEE_INT_RCP_SQRT_CONST_NR0 = 0x5F3759DF;
const int IEEE_INT_RCP_SQRT_CONST_NR1 = 0x5F375A86;
const int IEEE_INT_RCP_SQRT_CONST_NR2 = 0x5F375A86;

// Approximate guess using integer float arithmetics based on IEEE floating point standard
float rcpSqrtIEEEIntApproximation(float inX, int inRcpSqrtConst)
{
	int x = floatBitsToInt(inX);
	x = inRcpSqrtConst - (x >> 1);
	return intBitsToFloat(x);
}

float rcpSqrtNewtonRaphson(float inXHalf, float inRcpX)
{
	return inRcpX * (-inXHalf * (inRcpX * inRcpX) + 1.5f);
}

//
// Using 0 Newton Raphson iterations
// Relative error : ~3.4% over full
// Precise format : ~small float
// 2 ALU
//
float fastRcpSqrtNR0(float inX)
{
	float  xRcpSqrt = rcpSqrtIEEEIntApproximation(inX, IEEE_INT_RCP_SQRT_CONST_NR0);
	return xRcpSqrt;
}

//
// Using 1 Newton Raphson iterations
// Relative error : ~0.2% over full
// Precise format : ~float float
// 6 ALU
//
float fastRcpSqrtNR1(float inX)
{
	float  xhalf = 0.5f * inX;
	float  xRcpSqrt = rcpSqrtIEEEIntApproximation(inX, IEEE_INT_RCP_SQRT_CONST_NR1);
	xRcpSqrt = rcpSqrtNewtonRaphson(xhalf, xRcpSqrt);
	return xRcpSqrt;
}

//
// Using 2 Newton Raphson iterations
// Relative error : ~4.6e-004%  over full
// Precise format : ~full float
// 9 ALU
//
float fastRcpSqrtNR2(float inX)
{
	float  xhalf = 0.5f * inX;
	float  xRcpSqrt = rcpSqrtIEEEIntApproximation(inX, IEEE_INT_RCP_SQRT_CONST_NR2);
	xRcpSqrt = rcpSqrtNewtonRaphson(xhalf, xRcpSqrt);
	xRcpSqrt = rcpSqrtNewtonRaphson(xhalf, xRcpSqrt);
	return xRcpSqrt;
}

// BRDF based on implementation in Filament.
// https://github.com/google/filament

float D_Ashikhmin(float linearRoughness, float NoH) {
    // Ashikhmin 2007, "Distribution-based BRDFs"
	float a2 = linearRoughness * linearRoughness;
	float cos2h = NoH * NoH;
	float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
	float sin4h = sin2h * sin2h;
	float cot2 = -cos2h / (a2 * sin2h);
	return 1.0 / (UNITY_PI * (4.0 * a2 + 1.0) * sin4h) * (4.0 * exp(cot2) + sin4h);
}

float D_Charlie(float linearRoughness, float NoH) {
    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    float invAlpha  = 1.0 / linearRoughness;
    float cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * UNITY_PI);
}

float V_Neubelt(float NoV, float NoL) {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return saturate(1.0 / (4.0 * (NoL + NoV - NoL * NoV)));
}

float D_GGX_Anisotropic(float NoH, vec3 h,
        vec3 t, vec3 b, float at, float ab) {
    float ToH = dot(t, h);
    float BoH = dot(b, h);
    float a2 = at * ab;
    vec3 v = vec3(ab * ToH, at * BoH, a2 * NoH);
    float v2 = dot(v, v);
    float w2 = a2 / v2;
    w2 = max(0.001, w2);
    return a2 * w2 * w2 * UNITY_INV_PI;
}

float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
        float ToL, float BoL, float NoV, float NoL) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float lambdaV = NoL * length(vec3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(vec3(at * ToL, ab * BoL, NoL));
    float v = 0.5 / (lambdaV + lambdaL + 1.0e-7);
    return v;
}

// From "From mobile to high-end PC: Achieving high quality anime style rendering on Unity"
vec3 ShiftTangent (vec3 T, vec3 N, float shift) 
{
	vec3 shiftedT = T + shift * N;
	return normalize(shiftedT);
}

float StrandSpecular(vec3 T, vec3 H, float exponent, float strength)
{
	//vec3 H = normalize(L+V);
	float dotTH = dot(T, H);
	float sinTH = sqrt(1.0-dotTH*dotTH);
	float dirAtten = smoothstep(-1.0, 0.0, dotTH);
	return dirAtten * pow(sinTH, exponent) * strength;
}

vec4 unity_SHAr = vec4(0.0);
vec4 unity_SHAg = vec4(0.0);
vec4 unity_SHAb = vec4(0.0);
vec4 unity_SHBr = vec4(0.0);
vec4 unity_SHBg = vec4(0.0);
vec4 unity_SHBb = vec4(0.0);
vec4 unity_SHC = vec4(0.0);

vec3 UnpackScaleNormal(vec4 normaltex, float scale) {
	vec2 normalxy = (normaltex.xy * 2.0 - vec2(1.0)) * scale;
	float z = sqrt(1.0 - clamp(dot(normalxy, normalxy), 0.0, 1.0));
	return vec3(normalxy, z);
}


// normal should be normalized, w=1.0
vec3 SHEvalLinearL0L1 (vec4 normal)
{
    vec3 x;

    // Linear (L1) + constant (L0) polynomial terms
    x.r = dot(unity_SHAr,normal);
    x.g = dot(unity_SHAg,normal);
    x.b = dot(unity_SHAb,normal);

    return x;
}

// normal should be normalized, w=1.0
vec3 SHEvalLinearL2 (vec4 normal)
{
    vec3 x1, x2;
    // 4 of the quadratic (L2) polynomials
    vec4 vB = normal.xyzz * normal.yzzx;
    x1.r = dot(unity_SHBr,vB);
	    x1.g = dot(unity_SHBg,vB);
    x1.b = dot(unity_SHBb,vB);

    // Final (5th) quadratic (L2) polynomial
    float vC = normal.x*normal.x - normal.y*normal.y;
    x2 = unity_SHC.rgb * vC;

    return x1 + x2;
}

// normal should be normalized, w=1.0
// output in active color space
vec3 ShadeSH9 (vec4 normal)
{
    // Linear + constant polynomial terms
    vec3 res = SHEvalLinearL0L1 (normal);

    // Quadratic polynomials
    res += SHEvalLinearL2 (normal);


    return res;
}


vec3 SimpleSH9(vec3 normal)
{
   return ShadeSH9(vec4(normal, 1));
}

//Get the maximum SH contribution
//synqark's Arktoon shader's shading method
vec3 GetSHLength ()
{
   vec3 x, x1;
   x.r = length(unity_SHAr);
   x.g = length(unity_SHAg);
   x.b = length(unity_SHAb);
   x1.r = length(unity_SHBr);
   x1.g = length(unity_SHBg);
   x1.b = length(unity_SHBb);
   return x + x1;
}

float getGreyscaleSH(vec3 normal)
{
    // Samples the SH in the weakest and strongest direction and uses the difference
    // to compress the SH result into 0-1 range.

    // However, for efficiency, we only get the direction from L1.
    vec3 ambientLightDirection = 
        Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz));

    // If this causes issues, it might be worth getting the min() of those two.
    //vec3 dd = vec3(unity_SHAr.w, unity_SHAg.w, unity_SHAb.w);
    vec3 dd = SimpleSH9(-ambientLightDirection);
    vec3 ee = SimpleSH9(normal);
    vec3 aa = GetSHLength(); // SHa and SHb

    ee = saturate3( (ee - dd) / (aa - dd));
    return abs(dot(ee, sRGB_Luminance));

    return dot(normal, ambientLightDirection);
}

float LerpOneTo(float b, float t)
{
    float oneMinusT = 1.0 - t;
    return oneMinusT + b * t;
}

vec3 LerpWhiteTo(vec3 b, float t)
{
    float oneMinusT = 1.0 - t;
    return vec3(oneMinusT, oneMinusT, oneMinusT) + b * t;
}

// Used for matcaps
vec3 applyBlendMode(int blendOp, vec3 a, vec3 b, float t)
{
    switch (blendOp) 
    {
        case 1: return a * LerpWhiteTo(b, t);
        case 2: return a + b * a * t;
        default: // case 0:
        	return a + b * t;
    }
}

vec2 getMatcapUVs(vec3 normal, vec3 viewDir)
{
    // Based on Masataka SUMI's implementation
    vec3 worldUp = vec3(0, 1, 0);
    vec3 worldViewUp = normalize(worldUp - viewDir * dot(viewDir, worldUp));
    vec3 worldViewRight = normalize(cross(viewDir, worldViewUp));
    return vec2(dot(worldViewRight, normal), dot(worldViewUp, normal)) * 0.5 + 0.5;
}

vec2 getMatcapUVsOriented(vec3 normal, vec3 viewDir, vec3 upDir)
{
    // Based on Masataka SUMI's implementation
    vec3 worldViewUp = normalize(upDir - viewDir * dot(viewDir, upDir));
    vec3 worldViewRight = normalize(cross(viewDir, worldViewUp));
    return vec2(dot(worldViewRight, normal), dot(worldViewUp, normal)) * 0.5 + 0.5;
}

vec3 applyMatcap(sampler2D src, vec2 matcapUV, vec3 dst, vec3 light, int blendMode, float blendStrength)
{
    return applyBlendMode(blendMode, dst, texture(src, matcapUV).rgb * light, blendStrength);
}

// Stylish lighting helpers

float lerpstep( float a, float b, float t)
{
    return saturate( ( t - a ) / ( b - a ) );
}

float smootherstep(float a, float b, float t) 
{
    t = saturate( ( t - a ) / ( b - a ) );
    return t * t * t * (t * (t * 6. - 15.) + 10.);
}

float sharpenLighting (float inLight, float softness)
{
    vec2 lightStep = 0.5 + vec2(-1, 1) * fwidth(inLight);
    lightStep = mix(vec2(0.0, 1.0), lightStep, 1.0-softness);
    inLight = smoothstep(lightStep.x, lightStep.y, inLight);
    return inLight;
}

// By default, use smootherstep because it has the best visual appearance.
// But some functions might work better with lerpstep.
float simpleSharpen (float x, float width, float mid, int smoothnessMode) // smoothnessMode = 2
{
    vec2 dx = vec2(dFdx(x), dFdy(x));
    float rf = (dot(dx, dx)*2.0);
    width = max(width, rf);

    switch (smoothnessMode)
    {
        case 0: x = lerpstep(mid-width, mid, x); break;
        case 1: x = smoothstep(mid-width, mid, x); break;
        case 2: x = smootherstep(mid-width, mid, x); break;
    }

    return x;
}

// Note: Disney diffuse must be multiply by diffuseAlbedo / PI. This is done outside of this function.
float DisneyDiffuse(float NdotV, float NdotL, float LdotH, float perceptualRoughness)
{
    float fd90 = 0.5 + 2.0 * LdotH * LdotH * perceptualRoughness;
    // Two schlick fresnel term
    float lightScatter   = (1.0 + (fd90 - 1.0) * Pow5_f(1.0 - NdotL));
    float viewScatter    = (1.0 + (fd90 - 1.0) * Pow5_f(1.0 - NdotV));

    return lightScatter * viewScatter;
}


/* http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf */
float shEvaluateDiffuseL1Geomerics_local(float L0, vec3 L1, vec3 n)
{
	// average energy
	float R0 = L0;

	// avg direction of incoming light
	vec3 R1 = 0.5f * L1;

	// directional brightness
	float lenR1 = length(R1);

	// linear angle between normal and direction 0-1
	//float q = 0.5f * (1.0f + dot(R1 / lenR1, n));
	//float q = dot(R1 / lenR1, n) * 0.5 + 0.5;
	float q = dot(normalize(R1), n) * 0.5 + 0.5;
	q = saturate(q); // Thanks to ScruffyRuffles for the bug identity.

	// power for q
	// lerps from 1 (linear) to 3 (cubic) based on directionality
	float p = 1.0 + 2.0 * lenR1 / R0;

	// dynamic range constant
	// should vary between 4 (highly directional) and 0 (ambient)
	float a = (1.0 - lenR1 / R0) / (1.0 + lenR1 / R0);

	return R0 * (a + (1.0 - a) * (p + 1.0) * pow(q, p));
}

vec3 BetterSH9 (vec4 normal) {
	vec3 indirect;
	vec3 L0 = vec3(unity_SHAr.w, unity_SHAg.w, unity_SHAb.w);
	indirect.r = shEvaluateDiffuseL1Geomerics_local(L0.r, unity_SHAr.xyz, normal.xyz);
	indirect.g = shEvaluateDiffuseL1Geomerics_local(L0.g, unity_SHAg.xyz, normal.xyz);
	indirect.b = shEvaluateDiffuseL1Geomerics_local(L0.b, unity_SHAb.xyz, normal.xyz);
	indirect = max(vec3(0.0), indirect);
	indirect += SHEvalLinearL2(normal);
	return indirect;
}

// Ref: http://jcgt.org/published/0003/02/03/paper.pdf
float SmithJointGGXVisibilityTerm (float NdotL, float NdotV, float roughness)
{
    /*
    // Original formulation:
    //  lambda_v    = (-1 + sqrt(a2 * (1 - NdotL2) / NdotL2 + 1)) * 0.5f;
    //  lambda_l    = (-1 + sqrt(a2 * (1 - NdotV2) / NdotV2 + 1)) * 0.5f;
    //  G           = 1 / (1 + lambda_v + lambda_l);

    // Reorder code to be more optimal
    half a          = roughness;
    half a2         = a * a;

    half lambdaV    = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
    half lambdaL    = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);

    // Simplify visibility term: (2.0f * NdotL * NdotV) /  ((4.0f * NdotL * NdotV) * (lambda_v + lambda_l + 1e-5f));
    return 0.5f / (lambdaV + lambdaL + 1e-5f);  // This function is not intended to be running on Mobile,
                                                // therefore epsilon is smaller than can be represented by half
    */
    // Approximation of the above formulation (simplify the sqrt, not mathematically correct but close enough)
    float a = roughness;
    float lambdaV = NdotL * (NdotV * (1.0 - a) + a);
    float lambdaL = NdotV * (NdotL * (1.0 - a) + a);

    return 0.5 / (lambdaV + lambdaL + 1.0e-5);

}

float GGXTerm (float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = (NdotH * a2 - NdotH) * NdotH + 1.0; // 2 mad
    return UNITY_INV_PI * a2 / (d * d + 1.0e-7); // This function is not intended to be running on Mobile,
                                            // therefore epsilon is smaller than what can be represented by half
}

float OneMinusReflectivityFromMetallic(float metallic, vec4 colorSpaceDielectricSpec)
{
    // We'll need oneMinusReflectivity, so
    //   1-reflectivity = 1-lerp(dielectricSpec, 1, metallic) = lerp(1-dielectricSpec, 0, metallic)
    // store (1-dielectricSpec) in unity_ColorSpaceDielectricSpec.a, then
    //   1-reflectivity = lerp(alpha, 0, metallic) = alpha + metallic*(0 - alpha) =
    //                  = alpha - metallic * alpha
    float oneMinusDielectricSpec = colorSpaceDielectricSpec.a;
    return oneMinusDielectricSpec - metallic * oneMinusDielectricSpec;
}

vec3 PreMultiplyAlpha (vec3 diffColor, float alpha, float oneMinusReflectivity, out float outModifiedAlpha)
{
	const bool _ALPHAPREMULTIPLY_ON = true;
    if (_ALPHAPREMULTIPLY_ON) {
        // NOTE: shader relies on pre-multiply alpha-blend (_SrcBlend = One, _DstBlend = OneMinusSrcAlpha)

        // Transparency 'removes' from Diffuse component
        diffColor *= alpha;

        // Reflectivity 'removes' from the rest of components, including Transparency
        // outAlpha = 1-(1-alpha)*(1-reflectivity) = 1-(oneMinusReflectivity - alpha*oneMinusReflectivity) =
        //          = 1-oneMinusReflectivity + alpha*oneMinusReflectivity
        outModifiedAlpha = 1.0 - oneMinusReflectivity + alpha*oneMinusReflectivity;
    } else {
        outModifiedAlpha = alpha;
    }
    return diffColor;
}



	
//import "res://SCSS/Shaders/SCSS_Input.shader";

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
uniform sampler2D _Ramp;
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
    float attenuation, vec3 thickness, vec3 indirectLight) //  indirectLight = vec3(0.0)
{
    vec3 vSSLight = l.dir + normalDirection * _SSSDist; // Distortion
    vec3 vdotSS = vec3(pow(saturate(dot(viewDirection, -vSSLight)), _SSSPow)) 
        * _SSSIntensity; 
    
    return mix(1.0, attenuation, 1.0) //float(any(l.dir.xyz))) 
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
	float factor0 = saturate(simpleSharpen(x, width0, offset0, 2));

	float offset1 = _2nd_ShadeColor_Step * tone1.bias; 
	float width1  = _2nd_ShadeColor_Feather;
	float factor1 = saturate(simpleSharpen(x, width1, offset1, 2));

	vec3 final;
	final = mix(tone1.col, tone0.col, factor1);

	if (_CrosstoneToneSeparation == 0.0) 	final = mix(final, vec3(1.0), factor0) * albedo;
	if (_CrosstoneToneSeparation == 1.0) 	final = mix(final, albedo, factor0);
	
	x = factor0;
	
	return final;
}

float applyShadowLift(float baseLight, float occlusion)
{
	baseLight *= occlusion;
	baseLight = _ShadowLift + baseLight * (1.0-_ShadowLift);
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
		//NdotL = min(mix(shadeVal, NdotL, attenuation), NdotL);
		NdotL = mix(shadeVal*NdotL, NdotL, attenuation);
	} else {
		NdotL = min(NdotL * attenuation, NdotL);
		//NdotL = mix(0.5, NdotL, attenuation);
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
	directLighting = vec3(0.0);
	indirectLighting = vec3(0.0);
	switch (int(_LightingCalculationType))
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
vec3 calcDiffuseGI(vec3 albedo, SCSS_TonemapInput tone0, SCSS_TonemapInput tone1, float occlusion, float softness,
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

	vec3 indirectContribution;

	if (!SCSS_CROSSTONE) {
		ambientLight = applyShadowLift(ambientLight, occlusion);
		indirectContribution = sampleRampWithOptions(ambientLight, softness);
		indirectLighting = mix(indirectLighting, directLighting, tone0.col);
		indirectAverage = mix(indirectAverage, directLighting, tone0.col);
	}

	if (SCSS_CROSSTONE) {
		ambientLight *= occlusion;
		indirectAverage *= albedo;
		indirectContribution = sampleCrossToneLighting(ambientLight, tone0, tone1, albedo);
	}

	vec3 lightContribution;

	if (SCSS_CROSSTONE) {
		if (_CrosstoneToneSeparation == 0.0) lightContribution = 
		mix(indirectAverage,
		mix(indirectLighting, directLighting, indirectContribution),
		ambientLightSplitFactor) * albedo;

		if (_CrosstoneToneSeparation == 1.0) lightContribution = 
		mix(indirectAverage,
		directLighting*indirectContribution,
		ambientLightSplitFactor);
	}

	if (!SCSS_CROSSTONE) {
		lightContribution = 
		mix(indirectAverage,
		mix(indirectLighting, directLighting, indirectContribution),
		ambientLightSplitFactor) * albedo;
	}

	return lightContribution;
}

// For directional lights where attenuation is shadow.
vec3 calcDiffuseBase(vec3 albedo, SCSS_TonemapInput tone0, SCSS_TonemapInput tone1, float occlusion, float perceptualRoughness, 
	float attenuation, float softness, SCSS_LightParam d, SCSS_Light l)
{
	float remappedLight = getRemappedLight(perceptualRoughness, d);
	remappedLight = remappedLight * 0.5 + 0.5;

	remappedLight = applyAttenuation(remappedLight, attenuation);

	vec3 lightContribution;
	if (!SCSS_CROSSTONE) {
		remappedLight = applyShadowLift(remappedLight, occlusion);
		lightContribution = mix(tone0.col, vec3(1.0), sampleRampWithOptions(remappedLight, softness)) * albedo;
	}

	if (SCSS_CROSSTONE) {
		remappedLight *= occlusion;
		lightContribution = sampleCrossToneLighting(remappedLight, tone0, tone1, albedo);
	}

	lightContribution *= l.color;

	return lightContribution;	
}

// For point/spot lights where attenuation is shadow+attenuation.
vec3 calcDiffuseAdd(vec3 albedo, SCSS_TonemapInput tone0, SCSS_TonemapInput tone1, float occlusion, float perceptualRoughness, 
	float softness, SCSS_LightParam d, SCSS_Light l)
{
	float remappedLight = getRemappedLight(perceptualRoughness, d);
	remappedLight = remappedLight * 0.5 + 0.5;

	vec3 lightContribution;
	if (!SCSS_CROSSTONE) {
		remappedLight = applyShadowLift(remappedLight, occlusion);
		lightContribution = sampleRampWithOptions(remappedLight, softness);

		vec3 directLighting = l.color;
		vec3 indirectLighting = l.color * tone0.col;

		lightContribution = mix(indirectLighting, directLighting, lightContribution) * albedo;
	}

	if (SCSS_CROSSTONE) {
		lightContribution = sampleCrossToneLighting(remappedLight, tone0, tone1, albedo);
		lightContribution *= l.color;
	}

	return lightContribution;
}

void getSpecularVD(float roughness, SCSS_LightParam d, SCSS_Light l, VertexOutput i,
	out float V, out float D)
{
	V = 0.0; D = 0.0;

	switch(int(_SpecularType))
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
	    float anisotropy = abs(_Anisotropy);
	    float at = max(roughness * (1.0 + anisotropy), 0.002);
	    float ab = max(roughness * (1.0 - anisotropy), 0.002);

		//#if 0
	    //float TdotL = dot(i.tangentDir, l.dir);
	    //float BdotL = dot(i.bitangentDir, l.dir);
	    //float TdotV = dot(i.tangentDir, viewDir);
	    //float BdotV = dot(i.bitangentDir, l.dir);
		//
	    //// Accurate but probably expensive
		//V = V_SmithGGXCorrelated_Anisotropic (at, ab, TdotV, BdotV, TdotL, BdotL, d.NdotV, d.NdotL);
		//#else
		V = SmithJointGGXVisibilityTerm (d.NdotL, d.NdotV, roughness);
		//#endif
		// Temporary
	    D = D_GGX_Anisotropic(d.NdotH, d.halfDir, i.tangentDir, i.bitangentDir, at, ab);
	    break;
	}
	return;
}  

vec3 calcSpecularBaseInt(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	float attenuation, float occlusion, SCSS_LightParam d, SCSS_Light l, VertexOutput i, vec3 indirect_specular)
{
	
	float V = 0.0; float D = 0.0; 
	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	// "GGX with roughness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughness remapping."
	// This also fixes issues with the other specular types.
	roughness = max(roughness, 0.002);

	// d = saturate(d);
	d.NdotL = saturate(d.NdotL);
	d.NdotV = saturate(d.NdotV);
	d.LdotH = saturate(d.LdotH);
	d.NdotH = saturate(d.NdotH);

	getSpecularVD(roughness, d, l, i, /*out*/ V, /*out*/ D);

	float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later
	specularTerm = max(0.0, specularTerm * d.NdotL);

	if (!_SPECULARHIGHLIGHTS) {
    	specularTerm = 0.0;
	}

	float surfaceReduction = 1.0 / (roughness*roughness + 1.0);

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(notEqual(specColor, vec3(0.0))) ? 1.0 : 0.0;

	// gi =  GetUnityGI(l.color.rgb, l.dir, normal, 
	// 	d.viewDir, d.reflDir, attenuation, occlusion, perceptualRoughness, i.posWorld.xyz);

	float grazingTerm = saturate(smoothness + (1.0-oneMinusReflectivity));

	return
	specularTerm * (l.color.rgb * l.attenuation) * FresnelTerm(specColor, d.LdotH) +
	surfaceReduction * (indirect_specular) * FresnelLerp(specColor, vec3(grazingTerm), d.NdotV);
	
}

vec3 calcSpecularBase(SCSS_Input c, float perceptualRoughness, float attenuation,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularBaseInt(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, 
		perceptualRoughness, attenuation, c.occlusion, d, l, i, c.specular_light.rgb);
}

vec3 calcSpecularAddInt(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	if (!_SPECULARHIGHLIGHTS) {
		return vec3(0.0);
	}
	
	float V = 0.0; float D = 0.0;
	float roughness = PerceptualRoughnessToRoughness(perceptualRoughness);

	// "GGX with roughness to 0 would mean no specular at all, using max(roughness, 0.002) here to match HDrenderloop roughness remapping."
	// This also fixes issues with the other specular types.
	roughness = max(roughness, 0.002);

	// d = saturate(d);
	d.NdotL = saturate(d.NdotL);
	d.NdotV = saturate(d.NdotV);
	d.LdotH = saturate(d.LdotH);
	d.NdotH = saturate(d.NdotH);

	getSpecularVD(roughness, d, l, i, /*out*/ V, /*out*/ D);

	float specularTerm = V*D * UNITY_PI; // Torrance-Sparrow model, Fresnel is applied later
	specularTerm = max(0.0, specularTerm * d.NdotL);

    // To provide true Lambert lighting, we need to be able to kill specular completely.
    specularTerm *= any(notEqual(specColor, vec3(0.0))) ? 1.0 : 0.0;

	return
	specularTerm * l.color * FresnelTerm(specColor, d.LdotH);
	
}

vec3 calcSpecularAdd(SCSS_Input c, float perceptualRoughness,
	SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularAddInt(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, perceptualRoughness, d, l, i);
}

vec3 calcSpecularCelInt(vec3 specColor, float smoothness, vec3 normal, float oneMinusReflectivity, float perceptualRoughness,
	float attenuation, SCSS_LightParam d, SCSS_Light l, VertexOutput i, vec3 specular_light)
{
	if (_SpecularType == 4.0) {
		//  
		float spec = max(d.NdotH, 0.0);
		spec = pow(spec, (smoothness)*40.0) * _CelSpecularSteps;
		spec = sharpenLighting(fract(spec), _CelSpecularSoftness)+floor(spec);
    	spec = max(0.02,spec);
    	spec *= UNITY_PI / (_CelSpecularSteps); 

    	vec3 envLight = specular_light; //c == ) ?  : (unity_SpecCube0, normal, UNITY_SPECCUBE_LOD_STEPS);
		return (spec * specColor *  l.color) + (spec * specColor * envLight);
	}
	if (_SpecularType == 5.0) {
		vec3 strandTangent = (_Anisotropy < 0.0)
		? i.tangentDir
		: i.bitangentDir;
		strandTangent = mix(normal, strandTangent, abs(_Anisotropy));
		float exponent = smoothness;
		float spec  = StrandSpecular(strandTangent, d.halfDir, 
			exponent*80.0, 1.0 );
		float spec2 = StrandSpecular(strandTangent, d.halfDir, 
			exponent*40.0, 0.5 ); 
		spec  = sharpenLighting(fract(spec), _CelSpecularSoftness)+floor(spec);
		spec2 = sharpenLighting(fract(spec2), _CelSpecularSoftness)+floor(spec2);
		spec += spec2;
		
    	vec3 envLight = specular_light; // = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, normal, UNITY_SPECCUBE_LOD_STEPS);
		return (spec * specColor *  l.color) + (spec * specColor * envLight);
	}
	return vec3(0);
}

vec3 calcSpecularCel(SCSS_Input c, float perceptualRoughness, float attenuation, SCSS_LightParam d, SCSS_Light l, VertexOutput i)
{
	return calcSpecularCelInt(c.specColor, c.smoothness, c.normal, c.oneMinusReflectivity, perceptualRoughness, attenuation, d, l, i, c.specular_light.rgb);
}

vec3 SCSS_ShadeBase(SCSS_Input c, VertexOutput i, SCSS_Light l, float attenuation, bool is_directional_light)
{	
	vec3 finalColor;

	float isOutline = (_OutlineMode > 0.0 ? 1.0 : 0.0) * i.extraData.x;

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz, is_directional_light, l.dir);

	vec3 directLighting, indirectLighting;

	getDirectIndirectLighting(c.normal, /*out*/ directLighting, /*out*/ indirectLighting);

	finalColor  = calcDiffuseGI(c.albedo, c.tone0, c.tone1, c.occlusion, c.softness, indirectLighting, directLighting, d);
	finalColor += calcDiffuseBase(c.albedo, c.tone0, c.tone1, c.occlusion,
		c.perceptualRoughness, attenuation, c.softness, d, l);

	// FIXME: Enable only if SH is enabled.
	// Prepare fake light params for subsurface scattering.
	// SCSS_Light iL = l;
	// SCSS_LightParam iD = d;
	// iL.color = GetSHLength();
	// iL.dir = Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);
	// iD = initialiseLightParam(iL, c.normal, i.posWorld.xyz, is_directional_light);

	// Prepare fake light params for spec/fresnel which simulate specular.
	SCSS_Light fL = l;
	SCSS_LightParam fD = d;
	fL.color = attenuation * fL.color + GetSHLength();
	fL.dir = Unity_SafeNormalize(fL.dir + (unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz) * _LightSkew.xyz);
	fD = initialiseLightParam(fL, c.normal, i.posWorld.xyz, is_directional_light, l.dir);

	if (isOutline <= 0.0)
	{
		if (_SUBSURFACE) {
			if (is_directional_light) {
			finalColor += getSubsurfaceScatteringLight(l, c.normal, d.viewDir,
				attenuation, c.thickness, vec3(0.0)) * c.albedo;
			}
		// FIXME: Enable only if SH is enabled.
		// finalColor += getSubsurfaceScatteringLight(iL, c.normal, iD.viewDir,
		// 	1, c.thickness) * c.albedo;
		}

		if (_METALLICGLOSSMAP()) {
	    finalColor += calcSpecularBase(c, c.perceptualRoughness, attenuation, d, l, i);
	    }

	    if (_SPECGLOSSMAP()) {
    	finalColor += calcSpecularCel(c, c.perceptualRoughness, attenuation, fD, fL, i);
   		}
    }

    return finalColor;
}

vec3 SCSS_ShadeLight(SCSS_Input c, VertexOutput i, SCSS_Light l, float attenuation)
{
	vec3 finalColor;

	float isOutline = (_OutlineMode > 0.0 ? 1.0 : 0.0) * i.extraData.x;

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz, true, l.dir);

    finalColor = calcDiffuseAdd(c.albedo, c.tone0, c.tone1, c.occlusion, c.perceptualRoughness, c.softness, d, l);

	if (isOutline <= 0.0)
	{
		if (_SUBSURFACE) { 
		finalColor += c.albedo * getSubsurfaceScatteringLight(l, c.normal, d.viewDir,
			attenuation, c.thickness, c.tone0.col);
		}

		if (_METALLICGLOSSMAP()) {
    	finalColor += calcSpecularAdd(c, c.perceptualRoughness, d, l, i);
    	}

		if (_SPECGLOSSMAP()) {
		finalColor += calcSpecularCel(c, c.perceptualRoughness, attenuation, d, l, i);
		}
	}
	return finalColor;
}


vec3 SCSS_ApplyLighting(SCSS_Input c, VertexOutput i, vec4 texcoords, SCSS_Light l, bool is_base_pass, bool is_directional_light, float time)
{
	//UNITY_LIGHT_ATTENUATION(attenuation, i, i.posWorld.xyz);

	// FIXME.
	//if (SCSS_SCREEN_SHADOW_FILTER) && defined(USING_SHADOWS_UNITY) {
	//correctedScreenShadowsForMSAA(i._ShadowCoord, attenuation);
	//}


	float isOutline = (_OutlineMode > 0.0 ? 1.0 : 0.0) * i.extraData.x;

	// Lighting parameters
	//SCSS_Light l = MainLight(i.posWorld.xyz);
	//if (UNITY_PASS_FORWARDADD) && !defined(USING_DIRECTIONAL_LIGHT) {
	//l.dir = normalize(_WorldSpaceLightPos0.xyz - i.posWorld.xyz);
	//}

	SCSS_LightParam d = initialiseLightParam(l, c.normal, i.posWorld.xyz, is_directional_light, l.dir);

	// Generic lighting for matcaps/rimlighting. 
	// Currently matcaps are applied to albedo, so they don't need lighting. 
	vec3 effectLighting = l.color;
	if (is_base_pass) {
		//effectLighting *= attenuation;
		effectLighting += BetterSH9(vec4(0.0,  0.0, 0.0, 1.0));
	}

	// Apply matcap before specular effect.
	if (_UseMatcap >= 1.0 && isOutline <= 0.0)
	{
		vec2 matcapUV;
		if (_UseMatcap == 1.0) matcapUV = getMatcapUVsOriented(c.normal, d.viewDir, vec3(0, 1, 0));
		if (_UseMatcap == 2.0) matcapUV = getMatcapUVsOriented(c.normal, d.viewDir, i.bitangentDir.xyz);

		vec4 _MatcapMask_var = MatcapMask(i.uv0.xy);
		c.albedo = applyMatcap(_Matcap1, matcapUV, c.albedo.rgb, vec3(1.0), _Matcap1Blend, _Matcap1Strength * _MatcapMask_var.r);
		c.albedo = applyMatcap(_Matcap2, matcapUV, c.albedo.rgb, vec3(1.0), _Matcap2Blend, _Matcap2Strength * _MatcapMask_var.g);
		c.albedo = applyMatcap(_Matcap3, matcapUV, c.albedo.rgb, vec3(1.0), _Matcap3Blend, _Matcap3Strength * _MatcapMask_var.b);
		c.albedo = applyMatcap(_Matcap4, matcapUV, c.albedo.rgb, vec3(1.0), _Matcap4Blend, _Matcap4Strength * _MatcapMask_var.a);
	}

	vec3 finalColor = vec3(0.0); 

	float fresnelLightMaskBase = LerpOneTo((d.NdotH), _UseFresnelLightMask);
	float fresnelLightMask = 
		saturate(pow(saturate( fresnelLightMaskBase), _FresnelLightMask));
	float fresnelLightMaskInv = 
		saturate(pow(saturate(-fresnelLightMaskBase), _FresnelLightMask));
	
	// Lit
	if (_UseFresnel == 1.0 && isOutline <= 0.0)
	{
		vec3 sharpFresnel = sharpenLighting(d.rlPow4.y * c.rim.width * fresnelLightMask, 
			c.rim.power) * c.rim.tint * c.rim.alpha;
		sharpFresnel += sharpenLighting(d.rlPow4.y * c.rim.invWidth * fresnelLightMaskInv,
			c.rim.invPower) * c.rim.invTint * c.rim.invAlpha * _FresnelLightMask;
		c.albedo += c.albedo * sharpFresnel;
	}

	// AmbientAlt
	if (_UseFresnel == 3.0 && isOutline <= 0.0)
	{
		float sharpFresnel = sharpenLighting(d.rlPow4.y*c.rim.width, c.rim.power)
		 * c.rim.alpha;
		c.occlusion += saturate(sharpFresnel);
	}

	if (_UseFresnel == 4.0 && isOutline <= 0.0)
	{
		vec3 sharpFresnel = vec3(sharpenLighting(d.rlPow4.y * c.rim.width * fresnelLightMask, 
			c.rim.power));
		c.occlusion += saturate(sharpFresnel.r);
		sharpFresnel *= c.rim.tint * c.rim.alpha;
		sharpFresnel += sharpenLighting(d.rlPow4.y * c.rim.invWidth * fresnelLightMaskInv,
			c.rim.invPower) * c.rim.invTint * c.rim.invAlpha * _FresnelLightMask;
		c.albedo += c.albedo * sharpFresnel;
	}

	if (is_base_pass) {
	finalColor = SCSS_ShadeBase(c, i, l, l.attenuation, is_directional_light);
	} else {
	finalColor = SCSS_ShadeLight(c, i, l, l.attenuation);
	}

	// Proper cheap vertex lights. 
	//if (VERTEXLIGHT_ON) && !defined(SCSS_UNIMPORTANT_LIGHTS_FRAGMENT) {
	//finalColor += c.albedo * calcVertexLight(i.vertexLight, c.occlusion, c.tone, c.softness);
	//}

	// Ambient
	if (_UseFresnel == 2.0 && isOutline <= 0.0)
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
	// if (UNITY_PASS_FORWARDBASE) && defined(VERTEXLIGHT_ON) && defined(SCSS_UNIMPORTANT_LIGHTS_FRAGMENT) {
    // for (int num = 0; num < 4; num++) {
    // 	UNITY_BRANCH if ((unity_LightColor[num].r + unity_LightColor[num].g + unity_LightColor[num].b + i.vertexLight[num]) != 0.0)
    // 	{
    // 	l.color = unity_LightColor[num].rgb;
    // 	l.dir = normalize(vec3(unity_4LightPosX0[num], unity_4LightPosY0[num], unity_4LightPosZ0[num]) - i.posWorld.xyz);

	// 	finalColor += SCSS_ShadeLight(c, i, l, 1) *  i.vertexLight[num];	
    // 	}
    // };
	// }

	if (!is_base_pass) {
		finalColor *= l.attenuation;
	}

	finalColor *= _LightWrappingCompensationFactor;

	if (is_base_pass) {
	vec3 emission;
	vec4 emissionDetail = EmissionDetail(texcoords.zw, time);

	finalColor = max(vec3(0.0), finalColor - saturate3(vec3(1.0-emissionDetail.w)- (vec3(1.0)-c.emission)));
	emission = emissionDetail.rgb * c.emission * _EmissionColor.rgb;

	// Emissive c.rim. To restore masking behaviour, multiply by emissionMask.
	emission += _CustomFresnelColor.xyz * (pow(d.rlPow4.y, 1.0/(_CustomFresnelColor.w+0.0001)));

	emission *= (1.0-isOutline);
	finalColor += emission;
	}

	return finalColor;
}






















































































































































































































































































void vertex() {
	//o.pos = UnityObjectToClipPos(v.vertex);

	//o.normalDir= UnityObjectToWorldNormal(v.normal);
	//o.tangentDir = UnityObjectToWorldDir(v.tangent.xyz);
    //float sign = v.tangent.w * unity_WorldTransformParams.w;
	//o.bitangentDir = cross(o.normalDir, o .tangentDir) * sign;
	//vec4 objPos = mul(unity_ObjectToWorld, vec4(0, 0, 0, 1));
	posWorld = WORLD_MATRIX * vec4(VERTEX.xyz, 1.0);
	//o.vertex = v.vertex;

	// Extra data handling
	// X: Outline width | Y: Ramp softness
	// Z: Outline Z offset | 
	if (_VertexColorType == 2.0) 
	{
		extraData = COLOR;
		COLOR = vec4(1.0); // Reset
	} else {
		extraData = vec4(0.0, 0.0, 1.0, 1.0); 
		extraData.x = COLOR.a;
		COLOR = COLOR;
	}

	if (_OutlineMode != 0.0) {
		// TODO: calculate outline
		//#if defined(SCSS_USE_OUTLINE_TEXTURE)
		extraData.x *= OutlineMask(UV);
		//#endif

		extraData.x *= _outline_width * .01; // Apply outline width and convert to cm
		
		// Scale outlines relative to the distance from the camera. Outlines close up look ugly in VR because
		// they can have holes, being shells. This is also why it is clamped to not make them bigger.
		// That looks good at a distance, but not perfect. 
		extraData.x *= min(distance(posWorld.xyz,(CAMERA_MATRIX * vec4(0.0,0.0,0.0,1.0)).xyz)*4.0, 1.0); 
	}

	UV = AnimateTexcoords(UV, TIME);
	UV2 = UV2;
	NORMAL = NORMAL;

//#if defined(VERTEXLIGHT_ON)
//	o.vertexLight = VertexLightContribution(o.posWorld, o.normalDir);
//#endif
}

//VertexOutput vert_nogeom(appdata_full v) {
//	VertexOutput o = (VertexOutput)0;
//
//	o = vert(v);
//	
//	o.extraData.x = false;
//	return o;
//}

// [maxvertexcount(6)]
// void geom(triangle VertexOutput IN[3], inout TriangleStream<VertexOutput> tristream)
// {
//     #if defined(UNITY_REVERSED_Z)
//         const float far_clip_value_raw = 0.0;
//     #else
//         const float far_clip_value_raw = 1.0;
//     #endif

// 	// Generate base vertex
// 	[unroll]
// 	for (int ii = 0; ii < 3; ii++)
// 	{
// 		VertexOutput o = IN[ii];
// 		o.extraData.x = false;

// 		tristream.Append(o);
// 	}

// 	tristream.RestartStrip();

// 	// Generate outline vertex
// 	// If the outline triangle is too small, don't emit it.
// 	if ((IN[0].extraData.r + IN[1].extraData.r + IN[2].extraData.r) >= 1.e-9)
// 	{
// 		[unroll]
// 		for (int i = 2; i >= 0; i--)
// 		{
// 			VertexOutput o = IN[i];
// 			o.pos = UnityObjectToClipPos(o.vertex + normalize(o.normal) * o.extraData.r);

// 			// Possible future parameter depending on what people need
// 			float zPushLimit = lerp(far_clip_value_raw, o.pos.z, 0.9);
// 			o.pos.z = lerp(zPushLimit, o.pos.z, o.extraData.z);

// 			o.extraData.x = true;

// 			tristream.Append(o);
// 		}

// 		tristream.RestartStrip();
// 	}
// }

void combineLightDir(inout SCSS_Light mainLight, SCSS_Light newLight, vec3 normal, float shadow, float atten) {
	float dotp = max(dot(-normal, newLight.dir), 0.0);
	//newLight.intensity * 
	float intensity = newLight.color.r;//max(newLight.color.r, max(newLight.color.b, newLight.color.g));
	//mainLight.dir += intensity * dotp * newLight.attenuation * newLight.dir;
	mainLight.dir += intensity * newLight.dir; // 
}

void combineLight(inout SCSS_Light mainLight, SCSS_Light newLight, inout vec3 ambientContrib) {
	float dotp = -1.0 + 2.0 * step(0.0,max(dot(mainLight.dir, newLight.dir), -1.0));
	float intensity = newLight.intensity * max(newLight.color.r, max(newLight.color.b, newLight.color.g));
	vec3 newLightTerm = newLight.attenuation * newLight.color * newLight.intensity;
	mainLight.color += dotp * newLightTerm; //newLight.color; // newLightTerm;//dotp * newLightTerm;
	ambientContrib += max(vec3(0.0),(- dotp) * newLightTerm);
	//ambientContrib = vec3(0.0);
}

void fragment()
{
	// Initialize SH coefficients.
	LightmapCapture lc;
	if (false&&GET_LIGHTMAP_SH(lc)) {
		// TODO: Investigate multiplying by constants as in:
		// https://github.com/mrdoob/three.js/pull/16275/files
		vec3 constterm = SH_COEF(lc, uint(0)).rgb;
		unity_SHAr = vec4(SH_COEF(lc, uint(1)).rgb, constterm.r);
		unity_SHAg = vec4(SH_COEF(lc, uint(2)).rgb, constterm.g);
		unity_SHAb = vec4(SH_COEF(lc, uint(3)).rgb, constterm.b);
		vec3 shbX = SH_COEF(lc, uint(4)).rgb;
		vec3 shbY = SH_COEF(lc, uint(5)).rgb;
		vec3 shbZ = SH_COEF(lc, uint(6)).rgb;
		vec3 shbW = SH_COEF(lc, uint(7)).rgb;
		unity_SHBr = vec4(shbX.r, shbY.r, shbZ.r, shbW.r);
		unity_SHBg = vec4(shbX.g, shbY.g, shbZ.g, shbW.g);
		unity_SHBb = vec4(shbX.b, shbY.b, shbZ.b, shbW.b);
		unity_SHC = vec4(SH_COEF(lc, uint(8)).rgb,0.0);
	} else {
		// Indirect Light
	    vec4 reflection_accum;
	    vec4 ambient_accum;
		
		vec3 env_reflection_light = vec3(0.0);
		
		vec3 world_space_up = vec3(0.0,1.0,0.0);
		vec3 up_normal = mat3(INV_CAMERA_MATRIX) * world_space_up;

		vec3 ambient_light_up;
		vec3 diffuse_light_up;
		vec3 specular_light_up;
		reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);
		AMBIENT_PROCESS(VERTEX, up_normal, ROUGHNESS, SPECULAR, false, VIEW, vec2(0.0), ambient_light_up, diffuse_light_up, specular_light_up);
	    for (uint idx = uint(0); idx < REFLECTION_PROBE_COUNT(CLUSTER_CELL); idx++) {
			REFLECTION_PROCESS(CLUSTER_CELL, idx, VERTEX, up_normal, ROUGHNESS, ambient_light_up, specular_light_up, ambient_accum, reflection_accum);
	    }
	    if (ambient_accum.a > 0.0) {
			ambient_light_up = ambient_accum.rgb / ambient_accum.a;
	    }
		
		
		vec3 ambient_light_down;
		vec3 diffuse_light_down;
		vec3 specular_light_down;
		reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);
		AMBIENT_PROCESS(VERTEX, -up_normal, ROUGHNESS, SPECULAR, false, VIEW, vec2(0.0), ambient_light_down, diffuse_light_down, specular_light_down);
		for (uint idx = uint(0); idx < REFLECTION_PROBE_COUNT(CLUSTER_CELL); idx++) {
			REFLECTION_PROCESS(CLUSTER_CELL, idx, VERTEX, -up_normal, ROUGHNESS, ambient_light_down, specular_light_down, ambient_accum, reflection_accum);
		}
		if (ambient_accum.a > 0.0) {
			ambient_light_down = ambient_accum.rgb / ambient_accum.a;
		}
		vec3 const_term = mix(ambient_light_down, ambient_light_up, 0.5);
		vec3 delta_term = 0.5*(ambient_light_up - ambient_light_down);

		unity_SHAr = vec4(world_space_up * delta_term.r, const_term.r);
		unity_SHAg = vec4(world_space_up * delta_term.g, const_term.g);
		unity_SHAb = vec4(world_space_up * delta_term.b, const_term.b);
	}

	float isOutline = (_OutlineMode > 0.0 ? 1.0 : 0.0) * extraData.x;
	//if (isOutline && !FRONT_FACING) discard;

	// Backface correction. If a polygon is facing away from the camera, it's lit incorrectly.
	// This will light it as though it is facing the camera (which it visually is), unless
	// it's part of an outline, in which case it's invalid and deleted. 
	//facing = backfaceInMirror()? !facing : facing; // Only needed for older Unity versions.
	//if (!facing) 
	//{
	//	i.normalDir *= -1;
	//	i.tangentDir *= -1;
	//	i.bitangentDir *= -1;
	//}

	// if (_UseInteriorOutline)
	// {
	//     isOutline = max(isOutline, 1-innerOutline(i));
	// }
	
    float outlineDarken = 1.0-isOutline;

	vec4 texcoords = TexCoords(UV, UV2);

	// Ideally, we should pass all input to lighting functions through the 
	// material parameter struct. But there are some things that are
	// optional. Review this at a later date...
	//i.uv0 = texcoords; 

	SCSS_Input c = SCSS_Input(
		vec3(0.0),0.0,vec3(0.0),0.0,vec3(0.0),0.0,0.0,0.0,0.0,vec3(0.0),vec3(0.0),
		SCSS_RimLightInput(0.0,0.0,vec3(0.0),0.0,0.0,0.0,vec3(0.0),0.0),
		SCSS_TonemapInput(vec3(0.0),0.0),
		SCSS_TonemapInput(vec3(0.0),0.0),
		vec3(0.0));

	float detailMask = DetailMask(texcoords.xy);

    vec3 normalTangent = NormalInTangentSpace(texcoords, detailMask);

	//if (_SpecularType != 0 )
	if (_SPECULAR()) {
		vec4 specGloss = SpecularGloss(texcoords, detailMask);

		c.specColor = specGloss.rgb;
		c.smoothness = specGloss.a;

		// Because specular behaves poorly on backfaces, disable specular on outlines. 
		c.specColor  *= outlineDarken;
		c.smoothness *= outlineDarken;
	}

	c.albedo = Albedo(texcoords);

	c.emission = Emission(texcoords.xy);

	if (!SCSS_CROSSTONE) {
		c.tone0 = Tonemap(texcoords.xy, c.occlusion);
		c.tone1 = SCSS_TonemapInput(vec3(0.0),0.0);
	}

	if (SCSS_CROSSTONE) {
		c.tone0 = Tonemap1st(texcoords.xy);
		c.tone1 = Tonemap2nd(texcoords.xy);
		c.occlusion = ShadingGradeMap(texcoords.xy);
	}

	for (uint idx = uint(0); idx < DECAL_COUNT(CLUSTER_CELL); idx++) {
		vec3 decal_emission;
		vec4 decal_albedo;
		vec4 decal_normal;
		vec4 decal_orm;
		vec3 uv_local;
		if (DECAL_PROCESS(CLUSTER_CELL, idx, VERTEX, dFdx(VERTEX), dFdy(VERTEX), NORMAL, uv_local, decal_albedo, decal_normal, decal_orm, decal_emission)) {
			if (SCSS_CROSSTONE && _CrosstoneToneSeparation == 1.0) {
				c.tone0.col.rgb = mix(c.tone0.col.rgb, decal_albedo.rgb, decal_albedo.a);
				c.tone1.col.rgb = mix(c.tone0.col.rgb, decal_albedo.rgb, decal_albedo.a);
			}
			c.albedo.rgb = mix(c.albedo.rgb, decal_albedo.rgb, decal_albedo.a);
			c.normal = normalize(mix(c.normal, decal_normal.rgb, decal_normal.a));
			//AO = mix(AO, decal_orm.r, decal_orm.a);
			c.smoothness = 1.0 - mix(1.0 - c.smoothness, decal_orm.g, decal_orm.a);
			//METALLIC = mix(METALLIC, decal_orm.b, decal_orm.a);
			c.emission += decal_emission;
		}
	}

	VertexOutput i;
	i.uv0 = texcoords.xy;
	i.posWorld = posWorld.xyz;
	i.extraData = extraData;
	i.normalDir = NORMAL;
	i.tangentDir = TANGENT;
	i.bitangentDir = BINORMAL;

    // Thanks, Xiexe!
    vec3 tspace0 = vec3(TANGENT.x, BINORMAL.x, NORMAL.x);
    vec3 tspace1 = vec3(TANGENT.y, BINORMAL.y, NORMAL.y);
    vec3 tspace2 = vec3(TANGENT.z, BINORMAL.z, NORMAL.z);

    vec3 calcedNormal;
    calcedNormal.x = dot(tspace0, normalTangent);
    calcedNormal.y = dot(tspace1, normalTangent);
    calcedNormal.z = dot(tspace2, normalTangent);
    
    calcedNormal = normalize(calcedNormal);
    vec3 bumpedTangent = (cross(BINORMAL, calcedNormal));
    vec3 bumpedBitangent = (cross(calcedNormal, bumpedTangent));

    // For our purposes, we'd like to keep the original normal in i, but warp the bi/tangents.
    c.normal = calcedNormal;
    i.tangentDir = bumpedTangent;
    i.bitangentDir = bumpedBitangent;

	// Vertex colour application. 
	switch (int(_VertexColorType))
	{
		case 2: 
		case 0: c.albedo = c.albedo * COLOR.rgb; break;
		case 1: c.albedo = mix(c.albedo, COLOR.rgb, isOutline); break;
	}
	
	c.softness = extraData.g;

	c.alpha = Alpha(texcoords.xy);

	c.alpha *= texture(_ColorMask, texcoords.xy).r;

	//if (ALPHAFUNCTION) {
	//    alphaFunction(c.alpha);
	//}
 
	vec3 baseCameraPos = (CAMERA_MATRIX * vec4(0.0,0.0,0.0,1.0)).xyz;
    vec3 baseWorldPos = (WORLD_MATRIX * vec4(0.0,0.0,0.0,1.0)).xyz;
	applyVanishing(baseCameraPos, baseWorldPos, c.alpha);
	
	if (applyAlphaClip(TIME, c.alpha, _Cutoff, SCREEN_UV.xy * VIEWPORT_SIZE, _AlphaSharp)) {
		discard;
	}

	c = applyOutline(c, isOutline);

    // Rim lighting parameters. 
	c.rim = initialiseRimParam();
	c.rim.power *= RimMask(texcoords.xy);
	c.rim.tint *= outlineDarken;

	// Scattering parameters
	c.thickness = Thickness(texcoords.xy);

	// Specular variable setup

	// Disable PBR dielectric setup in cel specular mode.
	vec4 colorSpaceDielectricSpec = _SPECGLOSSMAP() ? vec4(0, 0, 0, 1) : vec4(0.04, 0.04, 0.04, 1.0 - 0.04);

	if (_SPECULAR()) {

		// Specular energy converservation. From EnergyConservationBetweenDiffuseAndSpecular in UnityStandardUtils.cginc
		c.oneMinusReflectivity = 1.0 - max (max (c.specColor.r, c.specColor.g), c.specColor.b); // SpecularStrength

		if (_UseMetallic == 1.0)
		{
			// From DiffuseAndSpecularFromMetallic
			// FIXME: The code was written to pass in c.specColor as first arg, which truncates to red component.
			c.oneMinusReflectivity = OneMinusReflectivityFromMetallic(1.0 - c.oneMinusReflectivity, colorSpaceDielectricSpec);
			c.specColor = mix (colorSpaceDielectricSpec.rgb, c.albedo, c.specColor);
		}

		if (_UseEnergyConservation == 1.0)
		{
			c.albedo.xyz = c.albedo.xyz * (c.oneMinusReflectivity); 
			//c.tone[0].col = c.tone[0].col * (c.oneMinusReflectivity); 
			// As tonemap is multiplied against albedo, is this necessary?
		}

	    bumpedTangent = ShiftTangent(normalize(bumpedTangent), c.normal, c.smoothness);
	    bumpedBitangent = normalize(bumpedBitangent);
	}

	if (_METALLICGLOSSMAP()) {
		// Geometric Specular AA from HDRP
		c.smoothness = GeometricNormalFiltering(c.smoothness, i.normalDir.xyz, 0.25, 0.5);
	}

	if (_METALLICGLOSSMAP()) {
		// Perceptual roughness transformation. Without this, roughness handling is wrong.
		c.perceptualRoughness = SmoothnessToPerceptualRoughness(c.smoothness);
	} else {
		// Disable DisneyDiffuse for cel specular.
	}

	if (_SPECULAR()) {
		vec3 ambient_light;
		vec3 diffuse_light;
		vec3 specular_light;
		vec4 reflection_accum = vec4(0.0, 0.0, 0.0, 0.0);
		vec4 ambient_accum = vec4(0.0, 0.0, 0.0, 0.0);
		vec3 env_reflection_light = vec3(0.0);
		float specMagnitude = max(c.specColor.r, max(c.specColor.g, c.specColor.b));
		float roughness_val = _SPECGLOSSMAP() ? 1.0 : c.perceptualRoughness;
		AMBIENT_PROCESS(VERTEX, c.normal, roughness_val, specMagnitude, false, VIEW, vec2(0.0), ambient_light, diffuse_light, specular_light);
		for (uint idx = uint(0); idx < REFLECTION_PROBE_COUNT(CLUSTER_CELL); idx++) {
			REFLECTION_PROCESS(CLUSTER_CELL, idx, VERTEX, c.normal, roughness_val, ambient_light, specular_light, ambient_accum, reflection_accum);
		}
		if (ambient_accum.a > 0.0) {
			ambient_light = ambient_accum.rgb / ambient_accum.a;
		}
		if (reflection_accum.a > 0.0) {
			specular_light = reflection_accum.rgb / reflection_accum.a;
		}
		c.specular_light = specular_light;
	}

	//#if !(defined(_ALPHATEST_ON) || defined(_ALPHABLEND_ON) || defined(_ALPHAPREMULTIPLY_ON))
	//	c.alpha = 1.0;
	//#endif

    // When premultiplied mode is set, this will multiply the diffuse by the alpha component,
    // allowing to handle transparency in physically correct way - only diffuse component gets affected by alpha
    float outputAlpha;
    c.albedo = PreMultiplyAlpha (c.albedo, c.alpha, c.oneMinusReflectivity, outputAlpha);

	float dir_light_intensity = 0.0;
	uint main_dir_light = uint(99999);
	//for (uint idx = uint(0); idx < DIRECTIONAL_LIGHT_COUNT(); idx++) {
	//	DirectionalLightData ld = GET_DIRECTIONAL_LIGHT(idx);
	//	if (!SHOULD_RENDER_DIR_LIGHT(ld)) {
	//		continue;
	//	}
	//	vec3 thisLightColor = (GET_DIR_LIGHT_COLOR_SPECULAR(ld).rgb);
	//	float this_intensity = max(thisLightColor.r, max(thisLightColor.g, thisLightColor.b));
	//	if (this_intensity <= dir_light_intensity + 0.00001) {
	//		continue;
	//	}
	//	dir_light_intensity = this_intensity;
	//	main_dir_light = idx;
	//}

	vec3 finalColor = vec3(0.0);

	VertexOutput iWorldSpace = i;
	iWorldSpace.normalDir = mat3(CAMERA_MATRIX) * i.normalDir;
	iWorldSpace.tangentDir = mat3(CAMERA_MATRIX) * i.tangentDir;
	iWorldSpace.bitangentDir = mat3(CAMERA_MATRIX) * i.bitangentDir;
	vec3 oldCNormal = c.normal;
	c.normal = normalize(mat3(CAMERA_MATRIX) * c.normal);
	// "Base pass" lighting is done in world space.
	// This is because SH9 works in world space.
	// We run other ligthting in view space.

	SCSS_Light mainLight;
	if (dir_light_intensity > 0.0) {
		DirectionalLightData ld = GET_DIRECTIONAL_LIGHT(main_dir_light);
		vec3 shadow_color = vec3(1.0);
		float shadow;
		float transmittance_z = 1.0;
		DIRECTIONAL_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow_color, shadow, transmittance_z);

		mainLight.cameraPos = baseCameraPos;
		mainLight.color = (GET_DIR_LIGHT_COLOR_SPECULAR(ld).rgb) / UNITY_PI;
		mainLight.intensity = 1.0; // For now.
		mainLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_DIR_LIGHT_DIRECTION(ld).xyz);
		mainLight.attenuation = shadow;

	} else {
		mainLight.cameraPos = baseCameraPos;
		mainLight.color = vec3(0.0);
		mainLight.intensity = 1.0;
		mainLight.dir = vec3(0.0,0.0001,0.0); // already world space.
		mainLight.attenuation = 1.0;

	}

	//c.normal = oldCNormal; // Restore normals to view space.
	// Lighting handling
	finalColor = SCSS_ApplyLighting(c, iWorldSpace, texcoords, mainLight, true, (dir_light_intensity > 0.0), TIME);

	// Deliberately corrupt this data to make sure it's not being used.
	//unity_SHBr /= (length(NORMAL) - 1.0);
	//unity_SHBg /= (length(NORMAL) - 1.0);
	//unity_SHBb *= 1.0e+10;
	//unity_SHC *= 1.0e+10;

	for (uint idx = uint(0); idx < DIRECTIONAL_LIGHT_COUNT(); idx++) {
		if (idx == main_dir_light) {
			continue;
		}
		DirectionalLightData ld = GET_DIRECTIONAL_LIGHT(idx);
		if (!SHOULD_RENDER_DIR_LIGHT(ld)) {
			continue;
		}

		vec3 shadow_color = vec3(1.0);
		float shadow;
		float transmittance_z = 1.0;
		DIRECTIONAL_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow_color, shadow, transmittance_z);

		SCSS_Light dirLight;
		dirLight.cameraPos = vec3(0.0); // working in view space
		dirLight.color = (GET_DIR_LIGHT_COLOR_SPECULAR(ld).rgb) / UNITY_PI;
		dirLight.intensity = 1.0; // For now.
		dirLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_DIR_LIGHT_DIRECTION(ld).xyz);
		dirLight.attenuation = shadow;
		
		combineLightDir(mainLight, dirLight, c.normal, shadow, 1.0);

		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, dirLight, false, true, TIME);
	}
	for (uint idx = uint(0); idx < OMNI_LIGHT_COUNT(CLUSTER_CELL); idx++) {
		LightData ld = GET_OMNI_LIGHT(CLUSTER_CELL, idx);
		if (!SHOULD_RENDER_LIGHT(ld)) {
			continue;
		}

		float transmittance_z = 0.0;
		float shadow;
		vec3 shadow_color_enabled = GET_LIGHT_SHADOW_COLOR(ld).rgb;
		if (OMNI_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow, transmittance_z)) {
			//vec3 no_shadow = OMNI_PROJECTOR_PROCESS(ld, VERTEX, vertex_ddx, vertex_ddy);
			//shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);
		}

		SCSS_Light pointLight;
		pointLight.cameraPos = vec3(0.0); // working in view space
		pointLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_LIGHT_POSITION(ld).xyz - VERTEX);
		float atten = GET_OMNI_LIGHT_ATTENUATION_SIZE(ld, VERTEX).x;
		pointLight.color = GET_LIGHT_COLOR_SPECULAR(ld).rgb / UNITY_PI;
		pointLight.intensity = 1.0; // For now.
		pointLight.attenuation = shadow * atten;

		combineLightDir(mainLight, pointLight, c.normal, shadow, atten);
		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, pointLight, false, false, TIME);
	}
	for (uint idx = uint(0); idx < SPOT_LIGHT_COUNT(CLUSTER_CELL); idx++) {
		LightData ld = GET_SPOT_LIGHT(CLUSTER_CELL, idx);
		if (!SHOULD_RENDER_LIGHT(ld)) {
			continue;
		}
			
		float transmittance_z = 0.0;
		float shadow;
		vec3 shadow_color_enabled = GET_LIGHT_SHADOW_COLOR(ld).rgb;
		if (SPOT_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow, transmittance_z)) {
			//vec3 no_shadow = SPOT_PROJECTOR_PROCESS(ld, VERTEX, vertex_ddx, vertex_ddy);
			//shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);
		}

		SCSS_Light spotLight;
		spotLight.cameraPos = vec3(0.0); // working in view space
		spotLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_LIGHT_POSITION(ld).xyz - VERTEX);
		float atten = GET_SPOT_LIGHT_ATTENUATION_SIZE(ld, VERTEX).x;
		spotLight.color = GET_LIGHT_COLOR_SPECULAR(ld).rgb / UNITY_PI;
		spotLight.intensity = 1.0; // For now.
		spotLight.attenuation = shadow * atten;

		combineLightDir(mainLight, spotLight, c.normal, shadow, atten);
		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, spotLight, false, false, TIME);
	}
	

	if (any(notEqual(mainLight.dir, vec3(0.0)))) {
		mainLight.dir = normalize(mainLight.dir);
	}
	
	vec3 xambient = vec3(unity_SHAr.w, unity_SHAg.w, unity_SHAb.w);
	for (uint idx = uint(0); idx < DIRECTIONAL_LIGHT_COUNT(); idx++) {
		if (idx == main_dir_light) {
			continue;
		}
		DirectionalLightData ld = GET_DIRECTIONAL_LIGHT(idx);
		if (!SHOULD_RENDER_DIR_LIGHT(ld)) {
			continue;
		}

		vec3 shadow_color = vec3(1.0);
		float shadow;
		float transmittance_z = 1.0;
		DIRECTIONAL_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow_color, shadow, transmittance_z);

		SCSS_Light dirLight;
		dirLight.cameraPos = vec3(0.0); // working in view space
		dirLight.color = (GET_DIR_LIGHT_COLOR_SPECULAR(ld).rgb) / UNITY_PI;
		dirLight.intensity = 1.0; // For now.
		dirLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_DIR_LIGHT_DIRECTION(ld).xyz);
		dirLight.attenuation = shadow;
		
		combineLight(mainLight, dirLight, xambient);

		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, dirLight, false, true, TIME);
	}
	for (uint idx = uint(0); idx < OMNI_LIGHT_COUNT(CLUSTER_CELL); idx++) {
		LightData ld = GET_OMNI_LIGHT(CLUSTER_CELL, idx);
		if (!SHOULD_RENDER_LIGHT(ld)) {
			continue;
		}

		float transmittance_z = 0.0;
		float shadow;
		vec3 shadow_color_enabled = GET_LIGHT_SHADOW_COLOR(ld).rgb;
		if (OMNI_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow, transmittance_z)) {
			//vec3 no_shadow = OMNI_PROJECTOR_PROCESS(ld, VERTEX, vertex_ddx, vertex_ddy);
			//shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);
		}

		SCSS_Light pointLight;
		pointLight.cameraPos = vec3(0.0); // working in view space
		pointLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_LIGHT_POSITION(ld).xyz - VERTEX);
		float atten = GET_OMNI_LIGHT_ATTENUATION_SIZE(ld, VERTEX).x;
		pointLight.color = GET_LIGHT_COLOR_SPECULAR(ld).rgb / UNITY_PI;
		pointLight.intensity = 1.0; // For now.
		pointLight.attenuation = shadow * atten;

		combineLight(mainLight, pointLight, xambient);
		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, pointLight, false, false, TIME);
	}
	for (uint idx = uint(0); idx < SPOT_LIGHT_COUNT(CLUSTER_CELL); idx++) {
		LightData ld = GET_SPOT_LIGHT(CLUSTER_CELL, idx);
		if (!SHOULD_RENDER_LIGHT(ld)) {
			continue;
		}
			
		float transmittance_z = 0.0;
		float shadow;
		vec3 shadow_color_enabled = GET_LIGHT_SHADOW_COLOR(ld).rgb;
		if (SPOT_SHADOW_PROCESS(ld, VERTEX, NORMAL, shadow, transmittance_z)) {
			//vec3 no_shadow = SPOT_PROJECTOR_PROCESS(ld, VERTEX, vertex_ddx, vertex_ddy);
			//shadow_attenuation = mix(shadow_color_enabled.rgb, no_shadow, shadow);
		}

		SCSS_Light spotLight;
		spotLight.cameraPos = vec3(0.0); // working in view space
		spotLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(GET_LIGHT_POSITION(ld).xyz - VERTEX);
		float atten = GET_SPOT_LIGHT_ATTENUATION_SIZE(ld, VERTEX).x;
		spotLight.color = GET_LIGHT_COLOR_SPECULAR(ld).rgb / UNITY_PI;
		spotLight.intensity = 1.0; // For now.
		spotLight.attenuation = shadow * atten;

		combineLight(mainLight, spotLight, xambient);
		// Lighting handling
		//finalColor += SCSS_ApplyLighting(c, i, texcoords, spotLight, false, false, TIME);
	}
	
	unity_SHAr.w = xambient.r;
	unity_SHAg.w = xambient.g;
	unity_SHAb.w = xambient.b;
//	unity_SHAr.xyz = vec3(0.0);
//	unity_SHAb.xyz = vec3(0.0);
//	unity_SHAg.xyz = vec3(0.0);
//	unity_SHBr = vec4(0.0);
//	unity_SHBb = vec4(0.0);
//	unity_SHBg = vec4(0.0);
//	unity_SHC.xyz = vec3(0.0);
	
	
	
	
	
	bool has_dir = false;
	if (any(notEqual(mainLight.dir, vec3(0.0)))) {
		has_dir = true;
		
	}
	//mainLight.attenuation = 1.0;
	finalColor = SCSS_ApplyLighting(c, iWorldSpace, texcoords, mainLight, true, (has_dir), TIME);
	//finalColor = vec3(mainLight.attenuation * abs(mainLight.dir) * mainLight.color * mainLight.intensity);// * fract(mainLight.cameraPos));
	//finalColor = ((abs(c.normal)));

	vec3 lightmap = vec3(1.0,1.0,1.0);
	// #if defined(LIGHTMAP_ON)
	// 	lightmap = DecodeLightmap(UNITY_SAMPLE_TEX2D(unity_Lightmap, i.uv1 * unity_LightmapST.xy + unity_LightmapST.zw));
	// #endif
	// TODO: make sure we handle lightmapped case.

	vec4 finalRGBA = vec4(finalColor * lightmap, outputAlpha);
	// UNITY_APPLY_FOG(i.fogCoord, finalRGBA);
	//return finalRGBA;
	METALLIC = 0.0;
	ALBEDO = vec3(0.0);
	ALBEDO = finalRGBA.rgb;
	//ALBEDO = vec3(0.5,0.6,0.4);
	ROUGHNESS = 1.0;
	SPECULAR = 0.0;
	AMBIENT_LIGHT = vec3(1.0);
	DIFFUSE_LIGHT = vec3(0.0);
	SPECULAR_LIGHT = vec3(0.0);
	//ALBEDO = vec3(0.1*1.0);//vec3(abs(0.1 * normalize(c.normal)));
}

void light() {
	DIFFUSE_LIGHT = vec3(0.0);
	SPECULAR_LIGHT = vec3(1.0);
}











