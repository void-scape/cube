struct Camera {
    proj_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Light {
	proj_view: mat4x4<f32>,
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(1) @binding(0)
var<uniform> light: Light;

struct Chunk {
    position: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> chunk: Chunk;

// @group(3) @binding(0)
// var shadow_map: texture_depth_2d; 
// @group(3) @binding(1)
// var shadow_map_sampler: sampler_comparison; 

struct VertexInput {
    /// color_index | normal_index | z | y | x
    /// ---------------------------------------
    /// 1             3              5   8   5
    @location(0) packed: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) light_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) color: vec3<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
	let x = f32(in.packed & 0x3f);
	let y = f32((in.packed >> 6) & 0xff);
	let z = f32((in.packed >> 14) & 0x3f);
	let normal_index = (in.packed >> 20) & 7;
	let color_index = (in.packed >> 23) & 1;

	const NORMALS = array<vec3<f32>, 6>(
		vec3(1.0, 0.0, 0.0),
		vec3(-1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, -1.0, 0.0),
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, -1.0),
	);

	const COLORS = array<vec3<f32>, 2>(
		vec3(51.0 / 255.0, 45.0 / 255.0, 35.0 / 255.0),
		vec3(145.0 / 255.0, 137.0 / 255.0, 125.0 / 255.0),
	);

    var out: VertexOutput;
	let model_position = vec3(x, y, z) + chunk.position;
    out.clip_position = camera.proj_view * vec4(model_position, 1.0);

	let light_position = light.proj_view * vec4(model_position, 1.0);
	out.light_position = vec3(
	  light_position.xy * vec2(0.5, -0.5) + vec2(0.5),
	  light_position.z
	);

	out.world_position = model_position;
	out.world_normal = NORMALS[normal_index];
	out.color = COLORS[color_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let color = in.color;
	let normal = normalize(in.world_normal);

	let ambient_brightness = 0.15;
	let ambient = light.color * ambient_brightness;

	let light_dir = normalize(light.position - in.world_position);
	let diffuse_brightness = max(dot(normal, light_dir), 0.0);
	let diffuse = light.color * diffuse_brightness;

	let bias = max(0.02 * (1.0 - dot(normal, light_dir)), 0.002);
    // let shadow = textureSampleCompare(
    //   shadow_map, shadow_map_sampler,
    //   in.light_position.xy, in.light_position.z - bias
    // );

	// // TODO: This doesn't do shit about the aliasing.
	// var shadow = 0.0;
	// let texel_size = 1.0 / 1024.0;
	// for (var x = -1; x <= 1; x += 1) {
	// 	for (var y = -1; y <= 1; y += 1) {
	// 		shadow += textureSampleCompare(
	// 		    shadow_map, shadow_map_sampler,
	// 		    in.light_position.xy + vec2(f32(x), f32(y)), 
	// 			in.light_position.z - bias,
	// 		); 
	// 	}    
	// }
	// shadow /= 9.0;

	let buckets = 9.0;
	let light = ambient + diffuse;
	// let light = ambient + shadow * diffuse;
 	// let quantized_light = floor(light * buckets) / buckets;

    return vec4(light * color, 1.0);
}
