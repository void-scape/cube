use crate::{camera::Camera, render::byte_slice};
use glam::Mat4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CameraUniform {
    pub proj_view: Mat4,
}

pub struct CameraData {
    pub camera: CameraUniform,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CameraData {
    pub fn new(device: &wgpu::Device) -> Self {
        let camera = CameraUniform {
            proj_view: Mat4::IDENTITY,
        };

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        Self {
            camera,
            uniform,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, width: u32, height: u32, camera: &Camera) {
        self.camera.proj_view = camera
            .projection_matrix(width, height)
            .mul_mat4(&camera.view_matrix());
        queue.write_buffer(&self.uniform, 0, byte_slice(&[self.camera]));
    }
}
