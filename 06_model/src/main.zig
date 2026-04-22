const ash = @import("ash");
const std = @import("std");

const ModelRenderer = @import("model_renderer.zig").ModelRenderer;

const app_name = "glTF Model Viewer";
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

    var host_impl = ash.DesktopHost.init(allocator, window_width, window_height, app_name);
    var session = ash.Session.init(allocator, host_impl.asHost(), app_name, null);
    var renderer = ModelRenderer.init(allocator);

    try session.run(&renderer);
}
