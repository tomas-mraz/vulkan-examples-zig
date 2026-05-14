const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ash_dep = b.dependency("vulkan_ash", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "17_ray7_denoiser",
        .use_llvm = true,
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

    addSpirv(b, exe, "raygen_shader", "raygen.rgen");
    addSpirv(b, exe, "miss_shader", "miss.rmiss");
    addSpirv(b, exe, "closest_hit_shader", "closesthit.rchit");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the example");
    run_step.dependOn(&run_cmd.step);
}

fn addSpirv(b: *std.Build, exe: *std.Build.Step.Compile, module_name: []const u8, shader_basename: []const u8) void {
    const cmd = b.addSystemCommand(&.{
        "glslangValidator",
        "-V",
        "--target-env",
        "vulkan1.2",
    });
    cmd.addPrefixedDirectoryArg("-I", b.path("shaders"));
    cmd.addArg("-o");
    const spv_name = b.fmt("{s}.spv", .{shader_basename});
    const out = cmd.addOutputFileArg(spv_name);
    cmd.addFileArg(b.path(b.fmt("shaders/{s}", .{shader_basename})));
    exe.root_module.addAnonymousImport(module_name, .{
        .root_source_file = out,
    });
}
