const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Debug currently crashes the Zig 0.17-dev compiler for this dependency mix.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Prioritize performance, safety, or binary size",
    ) orelse .ReleaseSafe;

    const ash_dep = b.dependency("vulkan_ash", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "05_cube",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "ash", .module = ash_dep.module("ash") },
            },
        }),
    });

    const vert_cmd = b.addSystemCommand(&.{
        "glslangValidator",
        "-V",
        "-o",
    });
    const vert_spv = vert_cmd.addOutputFileArg("cube.vert.spv");
    vert_cmd.addFileArg(b.path("shaders/cube.vert"));
    exe.root_module.addAnonymousImport("vertex_shader", .{
        .root_source_file = vert_spv,
    });

    const frag_cmd = b.addSystemCommand(&.{
        "glslangValidator",
        "-V",
        "-o",
    });
    const frag_spv = frag_cmd.addOutputFileArg("cube.frag.spv");
    frag_cmd.addFileArg(b.path("shaders/cube.frag"));
    exe.root_module.addAnonymousImport("fragment_shader", .{
        .root_source_file = frag_spv,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}
