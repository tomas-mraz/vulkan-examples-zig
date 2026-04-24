const ash = @import("ash");
const std = @import("std");

const TriangleRenderer = @import("triangle_renderer.zig").TriangleRenderer;

const app_name = "Rotating Triangle";

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

    var host_impl = ash.DesktopHost.initFullscreen(allocator, app_name);
    var session = ash.Session.init(allocator, host_impl.asHost(), app_name, .{
        .swapchain_options = .{
            .present_mode = .mailbox,
        },
    });
    var renderer = TriangleRenderer.init(allocator, &host_impl);

    try session.run(&renderer);
}
