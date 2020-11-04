shader_type spatial;
render_mode cull_disabled;
import "res://SCSS/Shaders/SCSS_Core.shader";

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

SCSS_Input c = SCSS_Input(
	vec3(0.0),0.0,vec3(0.0),0.0,vec3(0.0),0.0,0.0,0.0,0.0,vec3(0.0),vec3(0.0),
	SCSS_RimLightInput(0.0,0.0,vec3(0.0),0.0,0.0,0.0,vec3(0.0),0.0),
	SCSS_TonemapInput(vec3(0.0),0.0),
	SCSS_TonemapInput(vec3(0.0),0.0),
	vec3(0.0));

VertexOutput i = VertexOutput(vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),vec4(0.0),vec2(0.0));

VertexOutput iWorldSpace;

vec4 texcoords;

void fragment()
{
	// Initialize SH coefficients.
	//LightmapCapture lc;
	//if (GET_LIGHTMAP_SH(lc)) {
	//	const float c1 = 0.429043;
	//	const float c2 = 0.511664;
	//	const float c3 = 0.743125;
	//	const float c4 = 0.886227;
	//	const float c5 = 0.247708;
	//	// multiplying by constants as in:
	//	// https://github.com/mrdoob/three.js/pull/16275/files
	//	vec3 constterm = c4 * SH_COEF(lc, uint(0)).rgb - c5 * SH_COEF(lc, uint(6)).rgb;
	//	vec3 shaX = 2.0 * c2 * SH_COEF(lc, uint(3)).rgb;
	//	vec3 shaY = 2.0 * c2 * SH_COEF(lc, uint(1)).rgb;
	//	vec3 shaZ = 2.0 * c2 * SH_COEF(lc, uint(2)).rgb;
	//	vec3 shbX = 2.0 * c1 * SH_COEF(lc, uint(4)).rgb;
	//	vec3 shbY = 2.0 * c1 * SH_COEF(lc, uint(5)).rgb;
	//	vec3 shbZ = c3 * SH_COEF(lc, uint(6)).rgb;
	//	vec3 shbW = 2.0 * c1 * SH_COEF(lc, uint(7)).rgb;
	//	vec3 shc = c1 * SH_COEF(lc, uint(8)).rgb;
	//	unity_SHAr = vec4(shaX.r, shaY.r, shaZ.r, constterm.r);
	//	unity_SHAg = vec4(shaX.g, shaY.g, shaZ.g, constterm.g);
	//	unity_SHAb = vec4(shaX.b, shaY.b, shaZ.b, constterm.b);
	//	unity_SHBr = vec4(shbX.r, shbY.r, shbZ.r, shbW.r);
	//	unity_SHBg = vec4(shbX.g, shbY.g, shbZ.g, shbW.g);
	//	unity_SHBb = vec4(shbX.b, shbY.b, shbZ.b, shbW.b);
	//	unity_SHC = vec4(shc, 0.0);
	//} else {
		// Indirect Light
	    vec4 reflection_accum;
	    vec4 ambient_accum;

		vec3 world_space_up = vec3(0.0,1.0,0.0);
		vec3 up_normal = mat3(INV_CAMERA_MATRIX) * world_space_up;

		vec3 ambient_light_up;
		vec3 diffuse_light_up;
		vec3 specular_light_up;
		AMBIENT_PROCESS(VERTEX, up_normal, ROUGHNESS, SPECULAR, METALLIC, vec2(0.0), vec4(0.0), vec4(0.0), ambient_light_up, diffuse_light_up, specular_light_up);

		vec3 ambient_light_down;
		vec3 diffuse_light_down;
		vec3 specular_light_down;
		AMBIENT_PROCESS(VERTEX, -up_normal, ROUGHNESS, SPECULAR, METALLIC, vec2(0.0), vec4(0.0), vec4(0.0), ambient_light_down, diffuse_light_down, specular_light_down);

		vec3 const_term = mix(ambient_light_down, ambient_light_up, 0.5);
		vec3 delta_term = 0.5*(ambient_light_up - ambient_light_down);

		unity_SHAr = vec4(world_space_up * delta_term.r, const_term.r);
		unity_SHAg = vec4(world_space_up * delta_term.g, const_term.g);
		unity_SHAb = vec4(world_space_up * delta_term.b, const_term.b);
	//}

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

	texcoords = TexCoords(UV, UV2);

	// Ideally, we should pass all input to lighting functions through the 
	// material parameter struct. But there are some things that are
	// optional. Review this at a later date...
	//i.uv0 = texcoords; 

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

	{
		float tmp0;
		float tmp1;
		float tmp2;
		vec3 tmp3;
		float roughness = 1.0 - c.smoothness;
		vec3 tmpNormal = c.normal;
		vec3 tmpAlbedo = c.albedo;
		vec3 tmpEmission = c.emission;
		float tmpOcclusion = c.occlusion;
		APPLY_DECALS(VERTEX, tmpNormal, tmpAlbedo, tmpEmission, tmpOcclusion, roughness, tmp1);
		c.normal = tmpNormal;
		c.albedo = tmpAlbedo;
		c.emission = tmpEmission;
		c.occlusion = tmpOcclusion;
		if (SCSS_CROSSTONE && _CrosstoneToneSeparation == 1.0) {
			tmpAlbedo = c.tone0.col;
			APPLY_DECALS(VERTEX, tmpNormal, tmpAlbedo, tmpEmission, tmp0, tmp1, tmp2);
			c.tone0.col = tmpAlbedo;
			tmpAlbedo = c.tone1.col;
			APPLY_DECALS(VERTEX, tmpNormal, tmpAlbedo, tmpEmission, tmp0, tmp1, tmp2);
			c.tone1.col = tmpAlbedo;
		}
		c.smoothness = 1.0 - roughness;
	}

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
		float specMagnitude = max(c.specColor.r, max(c.specColor.g, c.specColor.b));
		float roughness_val = _SPECGLOSSMAP() ? 1.0 : c.perceptualRoughness;
		AMBIENT_PROCESS(VERTEX, c.normal, roughness_val, specMagnitude, 0.0, vec2(0.0), vec4(0.0), vec4(0.0), ambient_light, diffuse_light, specular_light);

		c.specular_light = specular_light;
	}

	//#if !(defined(_ALPHATEST_ON) || defined(_ALPHABLEND_ON) || defined(_ALPHAPREMULTIPLY_ON))
	//	c.alpha = 1.0;
	//#endif

	// When premultiplied mode is set, this will multiply the diffuse by the alpha component,
	// allowing to handle transparency in physically correct way - only diffuse component gets affected by alpha
	float outputAlpha;
	c.albedo = PreMultiplyAlpha (c.albedo, c.alpha, c.oneMinusReflectivity, /*out*/ outputAlpha);


	iWorldSpace = i;
	iWorldSpace.normalDir = mat3(CAMERA_MATRIX) * i.normalDir;
	iWorldSpace.tangentDir = mat3(CAMERA_MATRIX) * i.tangentDir;
	iWorldSpace.bitangentDir = mat3(CAMERA_MATRIX) * i.bitangentDir;
	vec3 oldCNormal = c.normal;
	c.normal = mat3(CAMERA_MATRIX) * c.normal;
	// "Base pass" lighting is done in world space.
	// This is because SH9 works in world space.
	// We run other ligthting in view space.

	AMBIENT_LIGHT = vec3(0.0);

	if (!HAS_MAIN_LIGHT) {
		SCSS_Light dirLight;
		dirLight.cameraPos = baseCameraPos;
		dirLight.color = vec3(0.0);
		dirLight.intensity = 0.0;
		dirLight.dir = vec3(0.0,1.0,0.0); // already world space.
		dirLight.attenuation = 1.0;

		DIFFUSE_LIGHT = SCSS_ApplyLighting(c, iWorldSpace, texcoords, dirLight, true, false, TIME, SPECULAR_LIGHT);
	} else {
		DIFFUSE_LIGHT = vec3(0.0);
		SPECULAR_LIGHT = vec3(0.0);
	}

	// c.normal = oldCNormal; // Restore normals to view space.
	NORMAL = c.normal;

	vec4 emissionDetail = EmissionDetail(texcoords.zw, TIME);

	// ALPHA = outputAlpha;

	ALBEDO = c.albedo;

	ROUGHNESS = 1.0;
	SPECULAR = 0.0;
	METALLIC =  0.0;
}

void light() {
	vec3 baseCameraPos = (CAMERA_MATRIX * vec4(0.0,0.0,0.0,1.0)).xyz;
    vec3 baseWorldPos = (WORLD_MATRIX * vec4(0.0,0.0,0.0,1.0)).xyz;
	vec3 finalColor = vec3(0.0);
	vec3 specularColor = vec3(0.0);
	if (IS_MAIN_LIGHT) {
		SCSS_Light dirLight;
		dirLight.cameraPos = baseCameraPos;
		dirLight.color = (LIGHT_COLOR.rgb) / UNITY_PI;
		dirLight.intensity = 1.0; // For now.
		dirLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(LIGHT);
		dirLight.attenuation = length(vec3(SHADOW_ATTENUATION))/length(vec3(1.0));

		// Lighting handling
		finalColor = SCSS_ApplyLighting(c, iWorldSpace, texcoords, dirLight, true, true, TIME, specularColor);
	} else {

		SCSS_Light pointLight;
		pointLight.cameraPos = baseCameraPos;
		pointLight.dir = mat3(CAMERA_MATRIX) * Unity_SafeNormalize(LIGHT);
		pointLight.color = LIGHT_COLOR.rgb / UNITY_PI;
		pointLight.intensity = 1.0; // For now.
		pointLight.attenuation = length(vec3(SHADOW_ATTENUATION * ATTENUATION))/length(vec3(1.0));

		// Lighting handling
		vec3 shadowResult = SCSS_ApplyLighting(c, iWorldSpace, texcoords, pointLight, false, false, TIME, specularColor);
		if (any(notEqual(SHADOW_COLOR, vec3(0.0)))) {
			vec3 tmp;
			pointLight.attenuation = length(vec3(ATTENUATION))/length(vec3(1.0));
			vec3 nonshadowResult = SCSS_ApplyLighting(c, iWorldSpace, texcoords, pointLight, false, false, TIME, tmp);
			finalColor += PROJECTOR_COLOR * shadowResult + SHADOW_COLOR * (nonshadowResult - shadowResult);
		} else {
			finalColor += PROJECTOR_COLOR * shadowResult;
		}
	}
	DIFFUSE_LIGHT += finalColor;
	SPECULAR_LIGHT += specularColor;
}
