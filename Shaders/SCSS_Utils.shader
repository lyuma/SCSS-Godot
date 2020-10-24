shader_type spatial;

// Portability functions. Overloads not possible.
vec4 saturate4(vec4 x) {
    return clamp(x, vec4(0.0), vec4(1.0));
}

vec4 saturate3(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}

vec4 saturate2(vec4 x) {
    return clamp(x, vec2(0.0), vec2(1.0));
}

vec4 saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 Unity_SafeNormalize(vec3 x) {
    return all(equal(x, vec3(0.0))) ? x : normalize(x);
}

struct GlobalParameters {

    vec3 baseWorldPos; // This world pos = actual world space!

    // In Godot, "world" space = view space
    vec3 viewSpaceCameraPos;
    vec4 lightColor0;
    vec3 viewSpaceLightPos0;

};
GlobalParamters _global;

const bool _ALPHATEST_ON = true;

const vec3 sRGB_Luminance = vec3(0.2126, 0.7152, 0.0722);

struct SCSS_Light
{
    vec3 color;
    vec3 dir;
    float  intensity; 
};


SCSS_Light MainLight()
{
    SCSS_Light l;

    l.color = _LightColor0.rgb;
    l.intensity = _LightColor0.w;
    l.dir = Unity_SafeNormalize(_WorldSpaceLightPos0.xyz); 

    // Workaround for scenes with HDR off blowing out in VRchat.
    //#if !UNITY_HDR_ON && SCSS_CLAMP_IN_NON_HDR
    //    l.color = saturate3(l.color);
    //#endif

    return l;
}

SCSS_Light MainLight(vec3 worldPos)
{
    SCSS_Light l = MainLight();
    l.dir = Unity_SafeNormalize(UnityWorldSpaceLightDir(worldPos)); 
    return l;
}

float interleaved_gradient(vec2 uv : SV_POSITION) : SV_Target
{
	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	return fract(magic.z * fract(dot(uv, magic.xy)));
}

float Dither17(vec2 Pos, float FrameIndexMod4)
{
    // 3 scalar float ALU (1 mul, 2 mad, 1 fract)
    return fract(dot(vec3(Pos.xy, FrameIndexMod4), uint3(2, 7, 23) / 17.0f));
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
	#define steps 4
	// pos is screen pixel position in 0-res range
    // Calculated noised gray value
    float noised = (2./steps) * T(intensity(vec2(pos.xy))) + gray - (1./steps); 
    // Clamp to the number of gray levels we want
    return floor(steps * noised) / (steps-1.);
    #undef steps
}

// "R2" dithering -- end

void applyAlphaClip(inout float alpha, float cutoff, vec2 pos, bool sharpen)
{
    pos += _SinTime.x%4;
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
        clip (alpha);
    } //#endif
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
    vec2 c = max(0.0001, fwidth(p));
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

const int IEEE_INT_RCP_SQRT_CONST_NR0 = 0x5f3759df;
const int IEEE_INT_RCP_SQRT_CONST_NR1 = 0x5F375A86;
const int IEEE_INT_RCP_SQRT_CONST_NR2 = 0x5F375A86;

// Approximate guess using integer float arithmetics based on IEEE floating point standard
float rcpSqrtIEEEIntApproximation(float inX, const int inRcpSqrtConst)
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

float D_GGX_Anisotropic(float NoH, const vec3 h,
        const vec3 t, const vec3 b, float at, float ab) {
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
    float v = 0.5 / (lambdaV + lambdaL + 1e-7f);
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

//vec3 SimpleSH9(vec3 normal)
//{
//    return ShadeSH9(vec4(normal, 1));
//}

// Get the maximum SH contribution
// synqark's Arktoon shader's shading method
//vec3 GetSHLength ()
//{
//    vec3 x, x1;
//    x.r = length(unity_SHAr);
//    x.g = length(unity_SHAg);
//    x.b = length(unity_SHAb);
//    x1.r = length(unity_SHBr);
//    x1.g = length(unity_SHBg);
//    x1.b = length(unity_SHBb);
//    return x + x1;
//}

// vec3 SHEvalLinearL2(vec3 n)
// {
//     return SHEvalLinearL2(vec4(n, 1.0));
// }

// float getGreyscaleSH(vec3 normal)
// {
//     // Samples the SH in the weakest and strongest direction and uses the difference
//     // to compress the SH result into 0-1 range.

//     // However, for efficiency, we only get the direction from L1.
//     vec3 ambientLightDirection = 
//         Unity_SafeNormalize((unity_SHAr.xyz + unity_SHAg.xyz + unity_SHAb.xyz));

//     // If this causes issues, it might be worth getting the min() of those two.
//     //vec3 dd = vec3(unity_SHAr.w, unity_SHAg.w, unity_SHAb.w);
//     vec3 dd = SimpleSH9(-ambientLightDirection);
//     vec3 ee = SimpleSH9(normal);
//     vec3 aa = GetSHLength(); // SHa and SHb

//     ee = saturate( (ee - dd) / (aa - dd));
//     return abs(dot(ee, sRGB_Luminance));

//     return dot(normal, ambientLightDirection);
// }

// Used for matcaps
vec3 applyBlendMode(int blendOp, vec3 a, vec3 b, float t)
{
    switch (blendOp) 
    {
        default:
        case 0: return a + b * t;
        case 1: return a * LerpWhiteTo(b, t);
        case 2: return a + b * a * t;
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
    return applyBlendMode(blendMode, dst, texture(src, matcapUV) * light, blendStrength);
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
    lightStep = lerp(vec2(0.0, 1.0), lightStep, 1-softness);
    inLight = smoothstep(lightStep.x, lightStep.y, inLight);
    return inLight;
}

// By default, use smootherstep because it has the best visual appearance.
// But some functions might work better with lerpstep.
float simpleSharpen (float x, float width, float mid, const float smoothnessMode = 2)
{
    vec2 dx = vec2(ddx(x), ddy(x));
    float rf = (dot(dx, dx)*2);
    width = max(width, rf);

    [flatten]
    switch (smoothnessMode)
    {
        case 0: x = lerpstep(mid-width, mid, x); break;
        case 1: x = smoothstep(mid-width, mid, x); break;
        case 2: x = smootherstep(mid-width, mid, x); break;
    }

    return x;
}
