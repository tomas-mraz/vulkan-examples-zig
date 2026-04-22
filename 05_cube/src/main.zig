const ash = @import("ash");
const std = @import("std");

const CubeRenderer = @import("cube_renderer.zig").CubeRenderer;

const app_name = "VulkanCube";
const window_width = 500;
const window_height = 500;

fn errorCallback(error_code: ash.glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}", .{ error_code, description });
}

pub fn main() !void {
    ash.glfw.setErrorCallback(errorCallback);
    ash.setDebug(true);
    ash.setValidations(false);

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var host_impl = ash.DesktopHost.init(allocator, window_width, window_height, app_name);
    var session = ash.Session.init(allocator, host_impl.asHost(), app_name, null);
    var renderer = CubeRenderer.init(allocator);

    try session.run(&renderer);
}
