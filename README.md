![Silent's Cel Shading Shaders](https://cdn.discordapp.com/attachments/414634326995763201/758190255521398784/SCSS_header_1.png)

# SCSS for Godot

This is a highly experimental and unmaintained fork of an old version of SCSS for Godot. It will be abandoned and replaced with a pure GLSL port of SCSS for Godot 4.x, but if you want to play with SCSS today in Godot 3.x, this repository is for you.

**You must use Godot's 3.x branch with the `shader_improvements_meta_3.x` branch from https://github.com/lyuma/godot merged in.**

To add this into your Godot project, clone this repo into res://SCSS exactly (not SCSS-Godot). Make sure that the folder res://SCSS/Shaders exists and contains the shader files.

To create a material with SCSS, give it the `SCSS_Forward.shader` and at the bottom of the Material inspector, inside the **Reference** heading, assign the "`SCSS_Inspector.gd`" Script.

Finally, note that a lot of the defaults are poor. Some settings like rim lighting mask requires ticking multiple checkboxes.

Setting up Outline support is challenging: you have to duplicate the shader and add `render_mode cull_front` to the duplicate copy, then duplicate the material, use the outline version of the shader on the duplicate, then assign the duplicate copy into the Next Pass of the original material.

SCSS is designed to be PBR compatible, and indeed contains some unique PBR features not even available in Godot's SpatialMaterial: it supports both standard metallic with reflections and anisotropy, combined with multicolor specular. Godot's SpatialMaterial does not support these features.

# Original README:

Shaders for Unity for cel shading, designed to take Unity's lighting into account while also presenting materials in the best possible way. Featuring lots of features with good performance!
# [Want to know how to use this shader? Here's the manual!](https://gitlab.com/s-ilent/SCSS/wikis/Manual/Setting-Overview)
# [Can't find the Download link? Click here!](https://gitlab.com/s-ilent/SCSS/-/archive/master/SCSS-master.zip)
* After downloading, install the shader by moving the contents of the Assets folder into your project's Assets folder.

![Suitable for shade or shine!](https://cdn.discordapp.com/attachments/414634326995763201/758184322708275220/Crosstone_proto.jpg)

## Features include:
* **Customisable lighting**

  A shadow tone map system is integrated, which allows for true anime-style material shade colouring and light bias. 
  Provides a light ramp system that are integrated seamlessly into lighting. 
  Or, use the Crosstone system and define multiple shadow tones.
  All integrated with Unity's lighting system!

* **NPR**

  SCSS contains a unique matcap system. You can combine multiple blend modes and multiple matcaps. They can be anchored in world or tangent space, stopping them from shifting with head movement in VR. 
  Customisable ambient and emissive rim lights are also provided for shine effects. 
  Cel-shaded specular gives you a stylised shiny highlight.

* **PBR**

  Contains metalness and smoothness functionality accurate to Unity's Standard shader. You can combine a cel-shaded material with realistic metal and gloss. 
  Detail maps are supported, allowing you to give materials a realistic fine texture close-up.

* **Outlines and control**

  The outline system is optimised for VR, with outlines that reduce size based on camera proximity to avoid models breaking up at close inspection. And outline size can be finely controlled using the vertex colour channels. 

* **Advanced Options**

  Many advanced options for blend mode and more. Provides support for using premutiplied transparency,   which allows for glossy transparent objects that naturally fit into their surroundings.

![Too Much Preview](https://cdn.discordapp.com/attachments/414634326995763201/694118872110071880/screen_10328x5640_2020-03-30_19-58-06.png.jpg)

# [For more details, please check the setting overview!](https://gitlab.com/s-ilent/SCSS/wikis/Manual/Setting-Overview)

Tested with Unity 2018.4.20f1 LTS.

For support, contact me on Discord or Twitter.

![Silent＃0264](https://files.catbox.moe/lv2mdh.png) 

![@Silent0264](https://files.catbox.moe/zma5gi.png)
