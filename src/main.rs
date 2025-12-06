fn main() {
    cube::platform::run(
        cube::Memory::default(),
        2560,
        1440,
        cube::handle_input,
        cube::update_and_render,
        cube::platform::debug_target(),
    );
}
