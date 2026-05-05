const ash = @import("ash");
const std = @import("std");
const vk = ash.vk;

const Ray4Renderer = @import("ray4_renderer.zig").Ray4Renderer;

const app_name = "Ray Tracing Reflections";
const window_width = 800;
const window_height = 600;

fn errorCallback(error_code: ash.glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}", .{ error_code, description });
}

pub fn main() !void {
    ash.glfw.setErrorCallback(errorCallback);
    ash.setDebug(false);
    ash.setValidations(false);

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var addr_features = vk.PhysicalDeviceBufferDeviceAddressFeatures{
        .buffer_device_address = .true,
    };
    var rt_features = vk.PhysicalDeviceRayTracingPipelineFeaturesKHR{
        .p_next = @ptrCast(&addr_features),
        .ray_tracing_pipeline = .true,
    };
    var as_features = vk.PhysicalDeviceAccelerationStructureFeaturesKHR{
        .p_next = @ptrCast(&rt_features),
        .acceleration_structure = .true,
    };
    var di_features = vk.PhysicalDeviceDescriptorIndexingFeatures{
        .p_next = @ptrCast(&as_features),
        .shader_sampled_image_array_non_uniform_indexing = .true,
        .descriptor_binding_variable_descriptor_count = .true,
        .runtime_descriptor_array = .true,
    };

    const enabled_features = vk.PhysicalDeviceFeatures{
        .shader_int_64 = .true,
        .shader_storage_image_read_without_format = .true,
        .shader_storage_image_write_without_format = .true,
    };

    const rt_extensions = [_][*:0]const u8{
        vk.extensions.khr_acceleration_structure.name,
        vk.extensions.khr_ray_tracing_pipeline.name,
        vk.extensions.khr_deferred_host_operations.name,
        vk.extensions.khr_buffer_device_address.name,
        vk.extensions.ext_descriptor_indexing.name,
        vk.extensions.khr_spirv_1_4.name,
        vk.extensions.khr_shader_float_controls.name,
    };

    var host_impl = ash.DesktopHost.init(allocator, window_width, window_height, app_name);
    var session = ash.Session.init(allocator, host_impl.asHost(), app_name, .{
        .device_options = .{
            .device_extensions = &rt_extensions,
            .p_next_chain = @ptrCast(&di_features),
            .enabled_features = &enabled_features,
            .api_version = vk.API_VERSION_1_2.toU32(),
        },
        .swapchain_options = .{
            .present_mode = .mailbox,
        },
    });
    var renderer = Ray4Renderer.init(allocator);

    try session.run(&renderer);
}
