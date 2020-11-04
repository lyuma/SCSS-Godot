tool
extends ShaderMaterial

var shader_float_to_int : Dictionary = {
	'_VertexColorType': true,
	'_CrosstoneToneSeparation': true,
	'_LightRampType': true,
	'_ShadowMaskType': true,
	'_OutlineMode': true,
	'_UseFresnel': true,
	'_SpecularType': true,
	'_UseMatcap': true,
	'_IndirectShadingType': true,
	'_LightingCalculationType': true,
}
var shader_int_to_bool : Dictionary = {
}
var shader_float_to_bool : Dictionary = {
	'_UseInteriorOutline': true,
	'_UseFresnelLightMask': true,
	'_FresnelLightMask': true,
	'_UseMetallic': true,
	'_UseEnergyConservation': true,
	'_UVSec': true,
	'_ThicknessMapInvert': true,
	'_UseVanishing': true,
	'_PixelSampleMode': true,
}
var shader_refresh_properties : Dictionary = {
	'_Color': true,
	'_Color_VALUE': true,
	'_EmissionColor': true,
	'_EmissionColor_VALUE': true,
	'SCSS_CROSSTONE': true,
	'_1st_ShadeColor': true,
	'_1st_ShadeColor_VALUE': true,
	'_2nd_ShadeColor': true,
	'_2nd_ShadeColor_VALUE': true,
	'_ShadowMaskColor': true,
	'_ShadowMaskColor_VALUE': true,
	'_OutlineMode': true,
	'_outline_color': true,
	'_outline_color_VALUE': true,
	'_UseFresnel': true,
	'_FresnelTint': true,
	'_FresnelTint_VALUE': true,
	'_FresnelTintInv': true,
	'_FresnelTintInv_VALUE': true,
	'_SpecularType': true,
	'_SpecColor': true,
	'_SpecColor_VALUE': true,
	'_UseMatcap': true,
	'_DETAIL': true,
	'_SUBSURFACE': true,
	'_SSSCol': true,
	'_SSSCol_VALUE': true,
	'_UseAnimation': true,
	'_UseVanishing': true,
	'_CustomFresnelColor': true,
	'_CustomFresnelColor_VALUE': true,
}
var shader_fake_params : Dictionary = {
	'SILENTS CEL SHADING SHADER': true,
	'EMISSION': true,
	'CROSSTONE SETTINGS': true,
	'LIGHT RAMP SETTINGS': true,
	'OUTLINE': true,
	'RIM': true,
	'SPECULAR': true,
	'MATCAP': true,
	'DETAIL': true,
	'SUBSURFACE SCATTERING': true,
	'ANIMATION': true,
	'VANISHING': true,
	'OTHER': true,
	'SYSTEM LIGHTING': true,
}
var shader_vec3_to_color : Dictionary = {
	'_EmissionColor': true,
	'_EmissionColor_VALUE': true,
	'_SpecColor': true,
	'_SpecColor_VALUE': true,
	'_SSSCol': true,
	'_SSSCol_VALUE': true,
}
var shader_vec4_to_color : Dictionary = {
	'_Color': true,
	'_Color_VALUE': true,
	'_1st_ShadeColor': true,
	'_1st_ShadeColor_VALUE': true,
	'_2nd_ShadeColor': true,
	'_2nd_ShadeColor_VALUE': true,
	'_ShadowMaskColor': true,
	'_ShadowMaskColor_VALUE': true,
	'_outline_color': true,
	'_outline_color_VALUE': true,
	'_FresnelTint': true,
	'_FresnelTint_VALUE': true,
	'_FresnelTintInv': true,
	'_FresnelTintInv_VALUE': true,
	'_CustomFresnelColor': true,
	'_CustomFresnelColor_VALUE': true,
}


func to_linear(c: Color) -> Color:
	return Color(
		c.r * (1.0 / 12.92) if c.r < 0.04045 else pow((c.r + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		c.g * (1.0 / 12.92) if c.g < 0.04045 else pow((c.g + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		c.b * (1.0 / 12.92) if c.b < 0.04045 else pow((c.b + 0.055) * (1.0 / (1 + 0.055)), 2.4),
		c.a)

func to_srgb(c: Color) -> Color:
	return Color(
		12.92 * c.r if c.r < 0.0031308 else (1.0 + 0.055) * pow(c.r, 1.0 / 2.4) - 0.055,
		12.92 * c.g if c.g < 0.0031308 else (1.0 + 0.055) * pow(c.g, 1.0 / 2.4) - 0.055,
		12.92 * c.b if c.b < 0.0031308 else (1.0 + 0.055) * pow(c.b, 1.0 / 2.4) - 0.055,
		c.a)

func _get(propertyname):
	var property = str(propertyname)
	if shader_fake_params.has(property):
		# return null if ret == false else 0
		return null
	var raw_value = false
	if property.ends_with('_VALUE') and (shader_vec3_to_color.has(property) or shader_vec4_to_color.has(property)):
		raw_value = true
		property = property.substr(0, property.length() - 6)
	if shader.has_param(property):
		var ret = get_shader_param(property)
		if shader_float_to_int.has(property):
			if typeof(ret) != typeof(1.234):
				print("property " + str(property) + " type " + str(typeof(ret)) + " ret " + str(ret))
			return int(ret)
		if shader_float_to_bool.has(property):
			return false if ret == 0.0 else true
		if shader_int_to_bool.has(property):
			return false if ret == 0 else true
		if not raw_value and shader_vec3_to_color.has(property):
			return to_srgb(Color(ret.x, ret.y, ret.z))
		if not raw_value and shader_vec4_to_color.has(property):
			return to_srgb(Color(ret.normal.x, ret.normal.y, ret.normal.z, ret.d))
		return ret

func _set(propertyname, value):
	var property = str(propertyname)
	if shader_fake_params.has(property):
		# value = false if (value == null) else true
		return true
	var raw_value = false
	if property.ends_with('_VALUE') and (shader_vec3_to_color.has(property) or shader_vec4_to_color.has(property)):
		raw_value = true
		property = property.substr(0, property.length() - 6)
	print("Set " + property + " to " + str(typeof(value)) + "," + str(value))
	if shader.has_param(property):
		if shader_float_to_int.has(property):
			value = float(value)
		if shader_float_to_bool.has(property):
			value = 1.0 if value else 0.0
		if shader_int_to_bool.has(property):
			value = 1 if value else 0
		if not raw_value and shader_vec3_to_color.has(property):
			value = to_linear(value)
			value = Vector3(value.r, value.g, value.b)
		if not raw_value and shader_vec4_to_color.has(property):
			value = to_linear(value)
			value = Plane(value.r, value.g, value.b, value.a)
		set_shader_param(property, value) # One can implement custom setter logic here
		if shader_refresh_properties.has(property):
			property_list_changed_notify()
		return true


func _get_property_list():
	var props: Array = []

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'SILENTS CEL SHADING SHADER',
	})
	props.push_back({
		'type': TYPE_OBJECT,
		'hint': PROPERTY_HINT_RESOURCE_TYPE,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'Texture',
		'name': '_MainTex',
	})
	props.push_back({
		'type': TYPE_PLANE,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_MainTex_ST',
	})
	props.push_back({
		'type': TYPE_COLOR,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_Color',
	})
	props.push_back({
		'type': TYPE_PLANE,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_Color_VALUE',
	})
	props.push_back({
		'type': TYPE_REAL,
		'hint': PROPERTY_HINT_RANGE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_Cutoff',
		'hint_string': '0,1',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_AlphaSharp',
	})
	props.push_back({
		'type': TYPE_OBJECT,
		'hint': PROPERTY_HINT_RESOURCE_TYPE,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'Texture',
		'name': '_ColorMask',
	})
	props.push_back({
		'type': TYPE_OBJECT,
		'hint': PROPERTY_HINT_RESOURCE_TYPE,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'Texture',
		'name': '_ClippingMask',
	})
	props.push_back({
		'type': TYPE_OBJECT,
		'hint': PROPERTY_HINT_RESOURCE_TYPE,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'Texture',
		'name': '_BumpMap',
	})
	props.push_back({
		'type': TYPE_REAL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_BumpScale',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'COLOR,OUTLINECOLOR,ADDITIONALDATA',
		'name': '_VertexColorType',
	})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'EMISSION',
	})
	props.push_back({
		'type': TYPE_OBJECT,
		'hint': PROPERTY_HINT_RESOURCE_TYPE,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'Texture',
		'name': '_EmissionMap',
	})
	props.push_back({
		'type': TYPE_COLOR,
		'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_EmissionColor',
	})
	props.push_back({
		'type': TYPE_VECTOR3,
		'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_EmissionColor_VALUE',
	})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'CROSSTONE SETTINGS' if _get('SCSS_CROSSTONE') else 'LIGHT RAMP SETTINGS',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': 'SCSS_CROSSTONE',
	})
	if _get('SCSS_CROSSTONE'):
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_1st_ShadeMap',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_1st_ShadeColor',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_1st_ShadeColor_VALUE',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_2nd_ShadeMap',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_2nd_ShadeColor',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_2nd_ShadeColor_VALUE',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_1st_ShadeColor_Step',
			'hint_string': '0,1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_1st_ShadeColor_Feather',
			'hint_string': '0.001, 1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_2nd_ShadeColor_Step',
			'hint_string': '0,1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_2nd_ShadeColor_Feather',
			'hint_string': '0.001, 1',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_ShadingGradeMap',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Tweak_ShadingGradeMapLevel',
			'hint_string': '-0.5, 0.5',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'COMBINED,SEPARATE',
			'name': '_CrosstoneToneSeparation',
		})
	else:
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'HORIZONTAL,VERTICAL,NONE',
			'name': '_LightRampType',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_Ramp',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'OCCLUSION,TONE,AUTO',
			'name': '_ShadowMaskType',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_ShadowMask',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_ShadowMaskColor',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_ShadowMaskColor_VALUE',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Shadow',
			'hint_string': '0,1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_ShadowLift',
			'hint_string': '-1,1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_IndirectLightingBoost',
			'hint_string': '0,1',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'OUTLINE',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'NONE,TINTED,COLORED',
		'name': '_OutlineMode',
	})
	if _get('_OutlineMode'):
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_OutlineMask',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_OutlineMask_ST',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_outline_width',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_outline_color',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_outline_color_VALUE',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_UseInteriorOutline',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_InteriorOutlineWidth',
			'hint_string': '0.0,1.0',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'RIM',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'DISABLE,LIT,AMBIENT,AMBIENTLIT',
		'name': '_UseFresnel',
	})
	if _get('_UseFresnel'):
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelWidth',
			'hint_string': '0,20',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelStrength',
			'hint_string': '0.01,0.9999',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelTint',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelTint_VALUE',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_UseFresnelLightMask',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelLightMask',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelTintInv',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelTintInv_VALUE',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelWidthInv',
			'hint_string': '0,20',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FresnelStrengthInv',
			'hint_string': '0.01, 0.9999',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'SPECULAR',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'DISABLE,STANDARD,CLOTH,ANISOTROPIC,CEL,CELSTRAND',
		'name': '_SpecularType',
	})
	if _get('_SpecularType'):
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SpecColor',
		})
		props.push_back({
			'type': TYPE_VECTOR3,
			'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SpecColor_VALUE',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_SpecGlossMap',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SpecGlossMap_ST',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_UseMetallic',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_UseEnergyConservation',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Smoothness',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_CelSpecularSoftness',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_CelSpecularSteps',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Anisotropy',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SPECULARHIGHLIGHTS',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'MATCAP',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'DISABLE,STANDARD,ANISOTROPIC',
		'name': '_UseMatcap',
	})
	if _get('_UseMatcap'):
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_MatcapMask',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_Matcap1',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Matcap1Strength',
			'hint_string': '0.0,2.0',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'ADDITIVE,MULTIPLY,MEDIAN',
			'name': '_Matcap1Blend',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_Matcap2',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Matcap2Strength',
			'hint_string': '0.0,2.0',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'ADDITIVE,MULTIPLY,MEDIAN,',
			'name': '_Matcap2Blend',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_Matcap3',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Matcap3Strength',
			'hint_string': '0.0,2.0',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'ADDITIVE,MULTIPLY,MEDIAN,,',
			'name': '_Matcap3Blend',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_Matcap4',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Matcap4Strength',
			'hint_string': '0.0,2.0',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_ENUM,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'ADDITIVE,MULTIPLY,MEDIAN,,,',
			'name': '_Matcap4Blend',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'DETAIL',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_DETAIL',
	})
	if _get('_DETAIL'):
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_DetailAlbedoMap',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_DetailAlbedoMap_ST',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_DetailAlbedoMapScale',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_DetailNormalMap',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_DetailNormalMapScale',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_SpecularDetailMask',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SpecularDetailStrength',
		})
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_DetailEmissionMap',
		})
		props.push_back({
			'type': TYPE_PLANE,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_EmissionDetailParams',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_UVSec',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'SUBSURFACE SCATTERING',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_SUBSURFACE',
	})
	if _get('_SUBSURFACE'):
		props.push_back({
			'type': TYPE_OBJECT,
			'hint': PROPERTY_HINT_RESOURCE_TYPE,
			'usage': PROPERTY_USAGE_EDITOR,
			'hint_string': 'Texture',
			'name': '_ThicknessMap',
		})
		props.push_back({
			'type': TYPE_BOOL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_ThicknessMapInvert',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_ThicknessMapPower',
			'hint_string': '0.01, 10.0',
		})
		props.push_back({
			'type': TYPE_COLOR,
			'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSCol',
		})
		props.push_back({
			'type': TYPE_VECTOR3,
			'hint': PROPERTY_HINT_COLOR_NO_ALPHA,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSCol_VALUE',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSIntensity',
			'hint_string': '0.0,10.0',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSPow',
			'hint_string': '0.01,10.0',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSDist',
			'hint_string': '0,10',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_RANGE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_SSSAmbient',
			'hint_string': '0,1',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'ANIMATION',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_UseAnimation',
	})
	if _get('_UseAnimation'):
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_AnimationSpeed',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_TotalFrames',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_FrameNumber',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Columns',
		})
		props.push_back({
			'type': TYPE_INT,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_Rows',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'VANISHING',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_UseVanishing',
	})
	if _get('_UseVanishing'):
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_VanishingStart',
		})
		props.push_back({
			'type': TYPE_REAL,
			'hint': PROPERTY_HINT_NONE,
			'usage': PROPERTY_USAGE_EDITOR,
			'name': '_VanishingEnd',
		})

	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'OTHER',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'TRANSPARENCY,SMOOTHNESS,CLIPPINGMASK',
		'name': '_AlbedoAlphaMode',
	})
	props.push_back({
		'type': TYPE_COLOR,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_CustomFresnelColor',
	})
	props.push_back({
		'type': TYPE_PLANE,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_CustomFresnelColor_VALUE',
	})
	props.push_back({
		'type': TYPE_BOOL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_PixelSampleMode',
	})
	props.push_back({
		'type': TYPE_NIL,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_CATEGORY,
		'name': 'SYSTEM LIGHTING',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'DYNAMIC,DIRECTIONAL,FLATTEN',
		'name': '_IndirectShadingType',
	})
	props.push_back({
		'type': TYPE_INT,
		'hint': PROPERTY_HINT_ENUM,
		'usage': PROPERTY_USAGE_EDITOR,
		'hint_string': 'ARKTOON,STANDARD,CUBED,DIRECTIONAL',
		'name': '_LightingCalculationType',
	})
	props.push_back({
		'type': TYPE_PLANE,
		'hint': PROPERTY_HINT_NONE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_LightSkew',
	})
	props.push_back({
		'type': TYPE_REAL,
		'hint': PROPERTY_HINT_RANGE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_DiffuseGeomShadowFactor',
		'hint_string': '0,1',
	})
	props.push_back({
		'type': TYPE_REAL,
		'hint': PROPERTY_HINT_RANGE,
		'usage': PROPERTY_USAGE_EDITOR,
		'name': '_LightWrappingCompensationFactor',
		'hint_string': '0.5, 1.0',
	})
	return props
