use crate::{
    camera::Camera,
    render::{camera::CameraData, chunk::ChunkData, light::LightData},
};

pub struct VoxelPipeline {
    depth_buffer: wgpu::TextureView,
    chunks: ChunkData,
    camera: CameraData,
    light: LightData,
}

impl VoxelPipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let depth_buffer_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel depth buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_buffer =
            depth_buffer_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let camera = CameraData::new(device);
        let light = LightData::new(device, surface_format, &camera);
        let chunks = ChunkData::new(device, surface_format, &camera, &light);

        Self {
            depth_buffer,
            chunks,
            camera,
            light,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        self.camera.update(queue, width, height, camera);
        self.chunks.update(device, camera);
        self.light
            .render(queue, encoder, view, &self.depth_buffer, &self.camera);
        self.chunks.render(
            device,
            queue,
            encoder,
            view,
            &self.depth_buffer,
            &self.camera,
            &self.light,
        );
    }
}
