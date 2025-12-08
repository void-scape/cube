use crate::{
    camera::Camera,
    platform::debug,
    render::{
        byte_slice,
        camera::CameraData,
        light::LightData,
        shadow::ShadowData,
        vert::{VOXEL_FACES, VOXEL_FACES_INDICES, VoxelVertex},
    },
};
use glam::{FloatExt, Vec2, Vec3};
use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU64,
};
use wgpu::util::DeviceExt;

pub const CHUNK_SIZE: usize = 32;
pub const VIEW_DISTANCE: usize = 4;

#[repr(C)]
pub struct ChunkUniform {
    position: Vec3,
    _pad: u32,
}

pub struct ChunkData {
    pub pipeline: wgpu::RenderPipeline,
    pub chunks: HashMap<(i64, i64), Chunk>,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: Vec<u8>,
}

impl ChunkData {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera: &CameraData,
        light: &LightData,
        shadow_map_bind_group_layout: &wgpu::BindGroupLayout,
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
                    shadow_map_bind_group_layout,
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

        Self {
            pipeline,
            chunks: Default::default(),
            uniform,
            bind_group,
            bind_group_layout,
            uniform_buffer,
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, camera: &Camera) {
        let (x, z) = (
            camera.translation.x as i64 / CHUNK_SIZE as i64,
            camera.translation.z as i64 / CHUNK_SIZE as i64,
        );
        let zrange = z - VIEW_DISTANCE as i64..=z + VIEW_DISTANCE as i64;
        let xrange = x - VIEW_DISTANCE as i64..=x + VIEW_DISTANCE as i64;

        self.chunks
            .retain(|(x, z), _| xrange.contains(x) && zrange.contains(z));

        let (dur, out) = debug::debug_time_millis(|| {
            std::thread::scope(|s| {
                let count = 2;
                let mut handles = Vec::with_capacity(count);
                'outer: for z in zrange.clone() {
                    for x in xrange.clone() {
                        if !self.chunks.contains_key(&(x, z)) {
                            handles.push(s.spawn(move || ((x, z), generate_voxels(x, z))));
                            // limit to 3 chunks a frame
                            if handles.len() >= count {
                                break 'outer;
                            }
                        }
                    }
                }

                let generated = handles.len();
                for handle in handles.into_iter() {
                    let ((x, z), (vertices, indices)) = handle.join().unwrap();
                    self.chunks
                        .insert((x, z), Chunk::new(device, &vertices, &indices));
                }

                generated
            })
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
        shadow: &ShadowData,
    ) {
        // upload chunk data to gpu
        let padding = device.limits().min_uniform_buffer_offset_alignment as usize
            - std::mem::size_of::<ChunkUniform>();

        self.uniform_buffer.clear();
        for (x, z) in self.chunks.keys() {
            self.uniform_buffer.extend(byte_slice(&[ChunkUniform {
                position: Vec3::new(
                    *x as f32 * CHUNK_SIZE as f32,
                    0.0,
                    *z as f32 * CHUNK_SIZE as f32,
                ),
                _pad: 0,
            }]));
            self.uniform_buffer.extend((0..padding).map(|_| 0));
        }
        queue.write_buffer(&self.uniform, 0, byte_slice(&self.uniform_buffer));

        // render to shadow map first
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow render pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow.shadow_map,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            render_pass.set_pipeline(&shadow.pipeline);
            render_pass.set_bind_group(0, &shadow.bind_group, &[]);

            let stride = device.limits().min_uniform_buffer_offset_alignment;
            for (i, chunk) in self.chunks.values().enumerate() {
                let offset = i as wgpu::DynamicOffset * stride;
                render_pass.set_bind_group(1, &self.bind_group, &[offset]);

                render_pass.set_vertex_buffer(0, chunk.vertices.slice(..));
                render_pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.indices_count, 0, 0..1);
            }
        }

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
            render_pass.set_bind_group(3, &shadow.shadow_map_bind_group, &[]);

            let stride = device.limits().min_uniform_buffer_offset_alignment;
            for (i, chunk) in self.chunks.values().enumerate() {
                let offset = i as wgpu::DynamicOffset * stride;
                render_pass.set_bind_group(2, &self.bind_group, &[offset]);

                render_pass.set_vertex_buffer(0, chunk.vertices.slice(..));
                render_pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.indices_count, 0, 0..1);
            }
        }
    }
}

pub struct Chunk {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub indices_count: u32,
}

impl Chunk {
    fn new(device: &wgpu::Device, vertices: &[VoxelVertex], indices: &[u32]) -> Self {
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vertex buffer"),
            contents: byte_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices_count = indices.len() as u32;
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk index buffer"),
            contents: byte_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertices,
            indices,
            indices_count,
        }
    }
}

fn generate_voxels(chunk_x: i64, chunk_z: i64) -> (Vec<VoxelVertex>, Vec<u32>) {
    let perlin_scale = 200;
    let noise_layers = [(1.5, 80.0), (3.0, 40.0), (8.0, 30.0)];

    let mut translations = Vec::new();

    let chunk_x = chunk_x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_z as f32 * CHUNK_SIZE as f32;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let uv = Vec2::new(
                (x as f32 + chunk_x) / perlin_scale as f32,
                (z as f32 + chunk_z) / perlin_scale as f32,
            );

            let mut surface = 0.0;
            for (uv_scale, weight) in noise_layers.iter() {
                surface += (perlin(uv * *uv_scale) * 0.5 + 0.5) * weight;
            }

            for y in 0..surface.round() as usize {
                translations.push((x as i32, y as i32, z as i32));
            }
        }
    }

    let hash: HashSet<(i32, i32, i32)> = translations.iter().copied().collect();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let neighbors = [
        (1, 0, 0, 3),
        (-1, 0, 0, 2),
        (0, 1, 0, 5),
        (0, -1, 0, 4),
        (0, 0, 1, 1),
        (0, 0, -1, 0),
    ];

    for (x, y, z) in translations.iter() {
        for (dx, dy, dz, face_idx) in neighbors.iter() {
            let neighbor_pos = (x + dx, y + dy, z + dz);

            if !hash.contains(&neighbor_pos) {
                let vertex_offset = vertices.len() as u32;
                for mut vertex in VOXEL_FACES[*face_idx].iter().cloned() {
                    debug_assert!(*x >= 0);
                    debug_assert!(*y >= 0);
                    debug_assert!(*z >= 0);
                    vertex.offset(*x as u32, *y as u32, *z as u32);
                    if *y > 50 {
                        vertex.set_color(1);
                    }
                    vertices.push(vertex);
                }
                for index in VOXEL_FACES_INDICES[*face_idx].iter() {
                    indices.push(vertex_offset + index);
                }
            }
        }
    }

    (vertices, indices)
}

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
