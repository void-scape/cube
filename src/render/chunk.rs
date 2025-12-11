use crate::{
    camera::Camera,
    platform::debug,
    render::{
        byte_slice, camera::CameraData, light::LightData, lines::ChunkLineRenderer,
        shadow::ShadowData, vert::VoxelVertex,
    },
};
use glam::{FloatExt, IVec3, UVec3, Vec2, Vec3, Vec4};
use std::{collections::HashMap, num::NonZeroU64};
use wgpu::util::DeviceExt;

pub const CHUNK_SIZE: usize = 32;
pub const VIEW_DISTANCE: usize = 22;
// pub const VIEW_DISTANCE: usize = 4;

#[repr(C)]
pub struct ChunkUniform {
    position: Vec3,
    _pad: u32,
}

pub struct ChunkData {
    pub pipeline: wgpu::RenderPipeline,
    pub chunks: HashMap<IVec3, ChunkBuffer>,
    pub visible: HashMap<IVec3, &'static ChunkBuffer>,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: Vec<u8>,
    // debug
    #[allow(unused)]
    lines: ChunkLineRenderer,
}

impl ChunkData {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera: &CameraData,
        light: &LightData,
        _shadow_map_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // TODO: The size of this buffer will depend on the devices `min_uniform_buffer_offset_alignment`.
        // Also, changing the number of chunks rendered would require resizing this buffer.
        let size = (VIEW_DISTANCE * VIEW_DISTANCE * VIEW_DISTANCE * 10) * 256;
        let uniform_buffer = Vec::with_capacity(size);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk uniform"),
            size: size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset: 0,
                    size: Some(
                        NonZeroU64::new(std::mem::size_of::<ChunkUniform>() as u64).unwrap(),
                    ),
                }),
            }],
        });

        let shader = wgpu::include_wgsl!("shaders/voxel.wgsl");
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel pipeline layout"),
                bind_group_layouts: &[
                    &camera.bind_group_layout,
                    &light.bind_group_layout,
                    &bind_group_layout,
                    // shadow_map_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let pipeline = crate::render::create_render_pipeline(
            device,
            &render_pipeline_layout,
            Some(surface_format),
            Some(wgpu::TextureFormat::Depth32Float),
            &[VoxelVertex::desc()],
            shader,
        );

        let lines = ChunkLineRenderer::new(device, surface_format, camera);

        Self {
            pipeline,
            chunks: HashMap::with_capacity(VIEW_DISTANCE * VIEW_DISTANCE),
            visible: HashMap::with_capacity(VIEW_DISTANCE * VIEW_DISTANCE),
            uniform,
            bind_group,
            bind_group_layout,
            uniform_buffer,
            //
            lines,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, camera: &Camera) {
        let (x, z) = (
            camera.translation.x as i32 / CHUNK_SIZE as i32,
            camera.translation.z as i32 / CHUNK_SIZE as i32,
        );
        let zrange = z - VIEW_DISTANCE as i32..=z + VIEW_DISTANCE as i32;
        let yrange = 0..4;
        let xrange = x - VIEW_DISTANCE as i32..=x + VIEW_DISTANCE as i32;

        // SAFETY: Make sure that visible doesn't point to any chunks.
        self.visible.clear();
        self.chunks.retain(|translation, _| {
            xrange.contains(&translation.x)
                && yrange.contains(&translation.y)
                && zrange.contains(&translation.z)
        });

        let (dur, out) = debug::debug_time_millis(|| {
            let mut generated = 0;
            for z in zrange.clone() {
                for y in yrange.clone() {
                    for x in xrange.clone() {
                        self.chunks.entry(IVec3::new(x, y, z)).or_insert_with(|| {
                            generated += 1;
                            let vertices = mesh_chunk(&generate_voxels(IVec3::new(x, y, z)));
                            ChunkBuffer::new(device, &vertices)
                        });
                    }
                }
            }
            generated
        });
        if out > 0 {
            println!("generated {out} chunks in {dur}ms");
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_buffer: &wgpu::TextureView,
        camera: &CameraData,
        light: &LightData,
        _shadow: &ShadowData,
    ) {
        let (_dur, _) = crate::platform::debug_time_micros(|| {
            // SAFETY: `self.visible` is cleared before writing to `self.chunks`,
            // therefore there will be no aliased mutable references.
            let visible = unsafe {
                std::mem::transmute::<
                    &mut HashMap<IVec3, &'static ChunkBuffer>,
                    &mut HashMap<IVec3, &ChunkBuffer>,
                >(&mut self.visible)
            };

            // upload chunk data to gpu
            let padding = device.limits().min_uniform_buffer_offset_alignment as usize
                - std::mem::size_of::<ChunkUniform>();

            // TODO: This needs to skip empty chunks.
            //
            // NOTE: Could definitely speed this up, but this function is only ~80us.
            for (chunk_translation, chunk) in self.chunks.iter() {
                let cs = CHUNK_SIZE as f32;
                let world_space = Vec4::new(
                    chunk_translation.x as f32 * cs,
                    chunk_translation.y as f32 * cs,
                    chunk_translation.z as f32 * cs,
                    1.0,
                );
                if [
                    Vec4::ZERO,
                    Vec4::new(cs, 0.0, 0.0, 0.0),
                    Vec4::new(cs, cs, 0.0, 0.0),
                    Vec4::new(0.0, cs, 0.0, 0.0),
                    //
                    Vec4::new(0.0, 0.0, cs, 0.0),
                    Vec4::new(cs, 0.0, cs, 0.0),
                    Vec4::new(cs, cs, cs, 0.0),
                    Vec4::new(0.0, cs, cs, 0.0),
                ]
                .into_iter()
                .any(|corner| {
                    let result = camera.camera.proj_view.mul_vec4(world_space + corner);
                    let viewspace = -1.0..=1.0;
                    viewspace.contains(&result.x)
                        && viewspace.contains(&result.y)
                        && viewspace.contains(&result.z)
                }) {
                    visible.insert(*chunk_translation, chunk);
                }
            }
            // println!("{}/{} chunks visible", visible.len(), self.chunks.len());

            self.uniform_buffer.clear();
            for chunk_translation in visible.keys() {
                self.uniform_buffer.extend(byte_slice(&[ChunkUniform {
                    position: chunk_translation.as_vec3() * CHUNK_SIZE as f32,
                    _pad: 0,
                }]));
                self.uniform_buffer.extend((0..padding).map(|_| 0));
            }
            queue.write_buffer(&self.uniform, 0, byte_slice(&self.uniform_buffer));

            // // render to shadow map first
            // {
            //     let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            //         label: Some("shadow render pass"),
            //         color_attachments: &[],
            //         depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            //             view: &shadow.shadow_map,
            //             depth_ops: Some(wgpu::Operations {
            //                 load: wgpu::LoadOp::Clear(1.0),
            //                 store: wgpu::StoreOp::Store,
            //             }),
            //             stencil_ops: None,
            //         }),
            //         ..Default::default()
            //     });
            //
            //     render_pass.set_pipeline(&shadow.pipeline);
            //     render_pass.set_bind_group(0, &shadow.bind_group, &[]);
            //
            //     let stride = device.limits().min_uniform_buffer_offset_alignment;
            //     for (i, chunk) in self.chunks.values().enumerate() {
            //         let offset = i as wgpu::DynamicOffset * stride;
            //         render_pass.set_bind_group(1, &self.bind_group, &[offset]);
            //
            //         render_pass.set_vertex_buffer(0, chunk.vertices.slice(..));
            //         render_pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
            //         render_pass.draw_indexed(0..chunk.indices_count, 0, 0..1);
            //     }
            // }

            // then do normal render pass
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("voxel render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_buffer,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &camera.bind_group, &[]);
                render_pass.set_bind_group(1, &light.bind_group, &[]);
                // render_pass.set_bind_group(3, &shadow.shadow_map_bind_group, &[]);

                let stride = device.limits().min_uniform_buffer_offset_alignment;
                for (i, chunk) in visible.values().enumerate() {
                    // for (i, chunk) in self.chunks.values().enumerate() {
                    let offset = i as wgpu::DynamicOffset * stride;
                    render_pass.set_bind_group(2, &self.bind_group, &[offset]);

                    if chunk.vertex_count > 0 {
                        render_pass.set_vertex_buffer(0, chunk.vertices.slice(..));
                        render_pass.draw(0..chunk.vertex_count, 0..1);
                    }
                }
            }

            // // debug
            // self.lines.render(
            //     queue,
            //     encoder,
            //     view,
            //     depth_buffer,
            //     camera,
            //     VIEW_DISTANCE as i32,
            // );
        });
        // println!("CPU render time: {dur}us");
    }
}

pub struct ChunkBuffer {
    vertices: wgpu::Buffer,
    vertex_count: u32,
}

impl ChunkBuffer {
    fn new(device: &wgpu::Device, vertices: &[VoxelVertex]) -> Self {
        let vertex_count = vertices.len() as u32;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vertex buffer"),
            contents: byte_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertices,
            vertex_count,
        }
    }
}

struct Chunk([Voxel; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]);

impl Default for Chunk {
    fn default() -> Self {
        Self([Voxel::default(); CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE])
    }
}

impl Chunk {
    #[track_caller]
    fn get(&self, xyz: UVec3) -> &Voxel {
        &self.0[Self::index(xyz)]
    }

    #[track_caller]
    fn get_mut(&mut self, xyz: UVec3) -> &mut Voxel {
        &mut self.0[Self::index(xyz)]
    }

    #[track_caller]
    fn index(xyz: UVec3) -> usize {
        debug_assert!(xyz.x < 32);
        debug_assert!(xyz.y < 32);
        debug_assert!(xyz.z < 32);
        const CS: u32 = CHUNK_SIZE as u32;
        (xyz.z * (CS * CS) + xyz.y * CS + xyz.x) as usize
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
enum Voxel {
    #[default]
    Air,
    Dirt,
    Stone,
}

fn generate_voxels(mut chunk_translation: IVec3) -> Chunk {
    // https://thebookofshaders.com/edit.php#11/2d-gnoise.frag
    fn perlin(st: Vec2) -> f32 {
        fn random2(st: Vec2) -> Vec2 {
            fn random(v: f32) -> f32 {
                -1.0 + 2.0 * (v.sin() * 43758.547).fract()
            }

            let st = Vec2::new(
                st.dot(Vec2::new(127.1, 311.7)),
                st.dot(Vec2::new(269.5, 183.3)),
            );
            Vec2::new(random(st.x), random(st.y))
        }

        let i = st.floor();
        let f = st - i;
        let u = f * f * (3.0 - 2.0 * f);

        let left = random2(i)
            .dot(f)
            .lerp(random2(i + Vec2::X).dot(f - Vec2::X), u.x);
        let right = random2(i + Vec2::Y)
            .dot(f - Vec2::Y)
            .lerp(random2(i + Vec2::ONE).dot(f - Vec2::ONE), u.x);
        left.lerp(right, u.y)
    }

    let perlin_scale = 200;
    let noise_layers = [(1.5, 80.0), (3.0, 40.0), (8.0, 30.0)];

    let mut chunk = Chunk::default();
    chunk_translation *= CHUNK_SIZE as i32;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let uv = Vec2::new(
                (x as f32 + chunk_translation.x as f32) / perlin_scale as f32,
                (z as f32 + chunk_translation.z as f32) / perlin_scale as f32,
            );

            let mut surface = 0.0;
            for (uv_scale, weight) in noise_layers.iter() {
                surface += (perlin(uv * *uv_scale) * 0.5 + 0.5) * weight;
            }

            let end = (surface.round() as i32 - chunk_translation.y).max(0);
            for y in 0..end {
                if y > 31 {
                    break;
                }

                let voxel = if y + chunk_translation.y > 50 {
                    Voxel::Stone
                } else {
                    Voxel::Dirt
                };
                *chunk.get_mut(UVec3::new(x as u32, y as u32, z as u32)) = voxel;
            }
        }
    }

    chunk
}

fn mesh_chunk(chunk: &Chunk) -> Vec<VoxelVertex> {
    // solid voxels for x, y, z axes
    let mut axis_cols = [[[0u32; CHUNK_SIZE]; CHUNK_SIZE]; 3];
    let mut col_face_masks = [[[0u32; CHUNK_SIZE]; CHUNK_SIZE]; 6];

    // populate columns with solid voxel data
    let mut voxels = chunk.0.iter();
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                // TODO: unchecked
                let voxel = voxels.next().unwrap();
                debug_assert_eq!(voxel, chunk.get(UVec3::new(x as u32, y as u32, z as u32)));
                if *voxel != Voxel::Air {
                    axis_cols[0][y][z] |= 1 << x as u32;
                    axis_cols[1][z][x] |= 1 << y as u32;
                    axis_cols[2][x][y] |= 1 << z as u32;
                }
            }
        }
    }

    // cull faces by checking the next slice on the axis
    for axis in 0..3 {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let lower = axis_cols[axis][y][x];
                let upper = if y + 1 < CHUNK_SIZE {
                    axis_cols[axis][y + 1][x]
                } else {
                    0
                };
                let below = if y > 0 { axis_cols[axis][y - 1][x] } else { 0 };
                col_face_masks[2 * axis][y][x] = lower & !upper;
                col_face_masks[2 * axis + 1][y][x] = lower & !below;
            }
        }
    }

    let mut vertices = Vec::with_capacity(32 * 32 * 4);
    let mut quads = Vec::with_capacity(1024);

    for (y, zx_faces) in col_face_masks[0].iter_mut().enumerate() {
        greedy_mesh(&mut quads, zx_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let xmin = min[1];
            let xmax = max[1];

            let zmin = min[0];
            let zmax = max[0];

            let y = y as u32;
            let normal = IVec3::Y;
            let v1 = VoxelVertex::new(xmax, y + 1, zmax, normal);
            let v2 = VoxelVertex::new(xmin, y + 1, zmax, normal);
            let v3 = VoxelVertex::new(xmin, y + 1, zmin, normal);
            let v4 = VoxelVertex::new(xmax, y + 1, zmin, normal);
            vertices.extend([v1, v4, v3, v3, v2, v1]);
        }
    }

    for (y, zx_faces) in col_face_masks[1].iter_mut().enumerate() {
        greedy_mesh(&mut quads, zx_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let xmin = min[1];
            let xmax = max[1];

            let zmin = min[0];
            let zmax = max[0];

            let y = y as u32;
            let normal = IVec3::NEG_Y;
            let v1 = VoxelVertex::new(xmax, y, zmax, normal);
            let v2 = VoxelVertex::new(xmin, y, zmax, normal);
            let v3 = VoxelVertex::new(xmin, y, zmin, normal);
            let v4 = VoxelVertex::new(xmax, y, zmin, normal);
            vertices.extend([v1, v2, v3, v3, v4, v1]);
        }
    }

    for (z, xy_faces) in col_face_masks[2].iter_mut().enumerate() {
        greedy_mesh(&mut quads, xy_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let ymin = min[1];
            let ymax = max[1];

            let xmin = min[0];
            let xmax = max[0];

            let z = z as u32;
            let normal = IVec3::Z;
            let v1 = VoxelVertex::new(xmax, ymax, z + 1, normal);
            let v2 = VoxelVertex::new(xmin, ymax, z + 1, normal);
            let v3 = VoxelVertex::new(xmin, ymin, z + 1, normal);
            let v4 = VoxelVertex::new(xmax, ymin, z + 1, normal);
            vertices.extend([v1, v2, v3, v3, v4, v1]);
        }
    }

    for (z, xy_faces) in col_face_masks[3].iter_mut().enumerate() {
        greedy_mesh(&mut quads, xy_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let ymin = min[1];
            let ymax = max[1];

            let xmin = min[0];
            let xmax = max[0];

            let z = z as u32;
            let normal = IVec3::NEG_Z;
            let v1 = VoxelVertex::new(xmax, ymax, z, normal);
            let v2 = VoxelVertex::new(xmin, ymax, z, normal);
            let v3 = VoxelVertex::new(xmin, ymin, z, normal);
            let v4 = VoxelVertex::new(xmax, ymin, z, normal);
            vertices.extend([v1, v4, v3, v3, v2, v1]);
        }
    }

    for (x, yz_faces) in col_face_masks[4].iter_mut().enumerate() {
        greedy_mesh(&mut quads, yz_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let zmin = min[1];
            let zmax = max[1];

            let ymin = min[0];
            let ymax = max[0];

            let x = x as u32;
            let normal = IVec3::X;
            let v1 = VoxelVertex::new(x + 1, ymax, zmax, normal);
            let v2 = VoxelVertex::new(x + 1, ymin, zmax, normal);
            let v3 = VoxelVertex::new(x + 1, ymin, zmin, normal);
            let v4 = VoxelVertex::new(x + 1, ymax, zmin, normal);
            vertices.extend([v1, v2, v3, v3, v4, v1]);
        }
    }

    for (x, yz_faces) in col_face_masks[5].iter_mut().enumerate() {
        greedy_mesh(&mut quads, yz_faces);
        for quad in quads.drain(..) {
            let min = quad.min;
            let max = quad.max;

            let zmin = min[1];
            let zmax = max[1];

            let ymin = min[0];
            let ymax = max[0];

            let x = x as u32;
            let normal = IVec3::NEG_X;
            let v1 = VoxelVertex::new(x, ymax, zmax, normal);
            let v2 = VoxelVertex::new(x, ymin, zmax, normal);
            let v3 = VoxelVertex::new(x, ymin, zmin, normal);
            let v4 = VoxelVertex::new(x, ymax, zmin, normal);
            vertices.extend([v1, v4, v3, v3, v2, v1]);
        }
    }

    vertices
}

#[derive(Debug)]
struct Quad {
    min: [u32; 2],
    max: [u32; 2],
}

// NOTE: Pre `greedy_mesh`, the game runs at ~400 fps with 8,100 total chunks. This
// measurement was taken without changing the angle of the camera. I expect this number
// to atleast double with the greedy mesher.
//
// UPDATE: It did not double, it is 1.65x, which is reasonable. It is still not perfectly
// efficient because it does not check neighboring chunks.
fn greedy_mesh(quads: &mut Vec<Quad>, faces: &mut [u32; 32]) {
    let len = 32;
    for x in 0..len {
        if faces[x] == 0 {
            continue;
        }

        for y in 0..len {
            let start = faces[x];
            let offset = (start >> y).trailing_zeros();
            if offset == 32 {
                continue;
            }
            let height = (start >> y >> offset).trailing_ones();
            let base_mask = 1u32.checked_shl(height).map(|v| v - 1).unwrap_or(!0);
            let quad_mask = !(base_mask << (offset + y as u32));
            let mut span = 0;
            for xoffset in 1..=len - x {
                if x + xoffset == len {
                    span = x + xoffset;
                    break;
                }
                let next = faces[x + xoffset];
                if ((next >> y >> offset) & base_mask) != base_mask {
                    span = x + xoffset;
                    break;
                }
                faces[x + xoffset] &= quad_mask;
            }
            faces[x] &= quad_mask;
            quads.push(Quad {
                min: [x as u32, offset + y as u32],
                max: [span as u32, offset + y as u32 + height],
            });
        }
    }
}
