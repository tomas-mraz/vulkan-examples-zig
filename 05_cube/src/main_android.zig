const std = @import("std");
const ash = @import("ash");

const CubeRenderer = @import("cube_renderer.zig").CubeRenderer;

const app_name = "VulkanCube";

pub export fn android_main(app: *ash.native_app_glue.android_app) callconv(.c) void {
    ash.setDebug(true);
    ash.setValidations(false);

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var host_impl = ash.AndroidHost.init(allocator, app);
    var session = ash.Session.init(allocator, host_impl.asHost(), app_name, null);
    var renderer = CubeRenderer.init(allocator);

    session.run(&renderer) catch |err| {
        std.log.err("session.run: {}", .{err});
    };
}
