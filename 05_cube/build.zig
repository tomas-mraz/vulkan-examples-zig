const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const abi = target.result.abi;
    const is_android = abi == .android or abi == .androideabi;

    const root_source = if (is_android)
        b.path("src/main_android.zig")
    else
        b.path("src/main_desktop.zig");

    const ash_dep = b.dependency("vulkan_ash", .{
        .target = target,
        .optimize = optimize,
    });

    const root_module = b.createModule(.{
        .root_source_file = root_source,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "ash", .module = ash_dep.module("ash") },
        },
    });

    var android_libc_file: ?std.Build.LazyPath = null;
    if (is_android) {
        const ndk = resolveAndroidNdk(b);
        const api_level = b.option(u32, "android-api", "Target Android API level") orelse 29;
        const arch_triple = androidArchTriple(target.result.cpu.arch);
        const sysroot = b.fmt("{s}/toolchains/llvm/prebuilt/linux-x86_64/sysroot", .{ndk});
        const include_dir = b.fmt("{s}/usr/include", .{sysroot});
        const arch_include_dir = b.fmt("{s}/usr/include/{s}", .{ sysroot, arch_triple });
        const lib_dir = b.fmt("{s}/usr/lib/{s}/{d}", .{ sysroot, arch_triple, api_level });
        root_module.addSystemIncludePath(.{ .cwd_relative = include_dir });
        root_module.addSystemIncludePath(.{ .cwd_relative = arch_include_dir });
        root_module.addLibraryPath(.{ .cwd_relative = lib_dir });

        // libc.txt redirects Zig from its bundled libc set to NDK Bionic.
        const libc_contents = b.fmt(
            "include_dir={s}\nsys_include_dir={s}\ncrt_dir={s}\nmsvc_lib_dir=\nkernel32_lib_dir=\ngcc_dir=\n",
            .{ include_dir, include_dir, lib_dir },
        );
        const wf = b.addWriteFiles();
        android_libc_file = wf.add("android-libc.txt", libc_contents);
    }

    const compile = if (is_android) b.addLibrary(.{
        .name = "05_cube",
        .linkage = .dynamic,
        .root_module = root_module,
        .use_llvm = true,
    }) else b.addExecutable(.{
        .name = "05_cube",
        .use_llvm = true,
        .root_module = root_module,
    });
    if (android_libc_file) |path| compile.setLibCFile(path);

    const vert_cmd = b.addSystemCommand(&.{
        "glslangValidator",
        "-V",
        "-o",
    });
    const vert_spv = vert_cmd.addOutputFileArg("cube.vert.spv");
    vert_cmd.addFileArg(b.path("shaders/cube.vert"));
    compile.root_module.addAnonymousImport("vertex_shader", .{
        .root_source_file = vert_spv,
    });

    const frag_cmd = b.addSystemCommand(&.{
        "glslangValidator",
        "-V",
        "-o",
    });
    const frag_spv = frag_cmd.addOutputFileArg("cube.frag.spv");
    frag_cmd.addFileArg(b.path("shaders/cube.frag"));
    compile.root_module.addAnonymousImport("fragment_shader", .{
        .root_source_file = frag_spv,
    });

    b.installArtifact(compile);

    if (!is_android) {
        const run_cmd = b.addRunArtifact(compile);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }

        const run_step = b.step("run", "Run the example");
        run_step.dependOn(&run_cmd.step);
    }
}

fn resolveAndroidNdk(b: *std.Build) []const u8 {
    if (b.option([]const u8, "android-ndk", "Path to the Android NDK root")) |path| return path;
    if (b.graph.environ_map.get("ANDROID_NDK_HOME")) |path| return path;
    if (b.graph.environ_map.get("ANDROID_NDK_ROOT")) |path| return path;
    @panic("Android target requires -Dandroid-ndk=<path>, ANDROID_NDK_HOME, or ANDROID_NDK_ROOT");
}

fn androidArchTriple(arch: std.Target.Cpu.Arch) []const u8 {
    return switch (arch) {
        .aarch64 => "aarch64-linux-android",
        .arm => "arm-linux-androideabi",
        .x86_64 => "x86_64-linux-android",
        .x86 => "i686-linux-android",
        else => @panic("unsupported Android architecture"),
    };
}
