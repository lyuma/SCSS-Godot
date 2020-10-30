//shader_type spatial;

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





































































































































































































































