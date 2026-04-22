# Vulkan Examples

This project is a collection of examples demonstrating the use of the Vulkan graphics API with the Go programming language.
The examples are organized progressively, from basic concepts to advanced techniques such as ray tracing.

Examples are using [Vulkan API Go binding](https://github.com/tomas-mraz/vulkan) and [Vulkan Ash framework](https://github.com/tomas-mraz/vulkan-ash).

Project URL: https://github.com/tomas-mraz/vulkan-examples

## Side notes

Conersion PNG image into RGBA format, compatible with Vulkan Image object, using ffmpeg `ffmpeg -v error -y -i ./gopher.png -f rawvideo -pix_fmt rgba out.rgba`


# List of Examples

| #  | Directory           | Description                                                                                |
|----|---------------------|--------------------------------------------------------------------------------------------|
| 01 | `01_version`        | Print Vulkan API and loader version                                                        |
| 02 | `02_info`           | Print GPU properties, extensions, layers                                                   |
| 03 | `03_draw`           | Static red triangle (pre-recorded command buffers)                                         |
| 04 | `04_triangle`       | Rotating blue triangle (push constants)                                                    |
| 05 | `05_cube`           | Textured rotating cube with gopher texture (uniform buffer, depth buffer, descriptor sets) |
| 06 | `06_model`          | Rotating 3D teapot model from glTF file (indexed drawing, depth buffer, directional light) |
| 07 | `07_model_textured` | Rotating textured teacup from GLB file (texture sampling, indexed drawing, depth buffer)   |
| 11 | `11_ray1`           | Ray-traced triangle (acceleration structures, ray tracing pipeline, storage image)         |
| 12 | `12_ray2`           | Ray-traced FlightHelmet glTF model with accumulation and shadows                           |
| 13 | `13_ray3`           | Ray-traced FlightHelmet glTF model with moving light and progressive accumulation          |


# Requirements

- **Zig 0.17+**
- **Vulkan SDK** (runtime, validation layers, shaders compiler)
- Vulkan runtime/loader (included in Vulkan SDK or in the drivers from graphic card)
- system variable **VULKAN_SDK**
- shared compilator glslangValidator (included in Vulkan SDK) or optionally linux package `glslang-tools`

Required 64bit operating system:
- Windows 10/11
- Linux with Wayland
- macOS 17+ with Metal 4

## macOS
Use Homebrew to install  
`mesa` - contains ICD Vulkan driver KosmicKrisp  
`vulkan-loader` - contains Vulkan loader vhich load KosmicKrisp driver

```bash
brew install vulkan-loader mesa
```

## Android
To support raytracing examples is needed hardware with one of theese chips:
- Snapdragon 8 Gen 2 a newer (Adreno 740+)
- MediaTek Dimensity 9200 a newer
- Samsung Exynos 2200 a newer (AMD RDNA2)
- Google Tensor G3 a newer


# Build

Each example has its own `Makefile`. To build and run in the project folder:
```bash
make run
```

To see available targets:
```bash
make help
```


# Dependencies

- vulkan-ash Vulkan framework
  - https://github.com/tomas-mraz/vulkan-ash.git
  - /home/tomas/git-osobni-github/vulkan-ash

- glfw binding
  - https://github.com/tomas-mraz/glfw-zig.git
  - /home/tomas/git-osobni-github/glfw-zig

- vulkan-zig Vulkan API binding
  - https://github.com/tomas-mraz/vulkan-zig.git
  - /home/tomas/git-osobni-github/vulkan-zig

- image library
  - https://github.com/zigimg/zigimg.git


# Related projects
- Sascha Willems Vulkan Examples in C++ https://github.com/SaschaWillems/Vulkan
- Official Khronos Group Vulkan Examples https://github.com/KhronosGroup/Vulkan-Samples


# Assets and licences

DiffuseTransmissionTeacup model using Diffuse Transmission extension.
Credit:
© 2023, Public domain. CC0 1.0 Universal
- Polyhaven.com, and Eric Chadwick for Models and Textures
