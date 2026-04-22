const builtin = @import("builtin");
const std = @import("std");
const vk = @import("vulkan");

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const Instance = vk.InstanceProxy;

var vulkan_loader: vk.PfnGetInstanceProcAddr = undefined;

pub fn main() !void {
    var lib = try openVulkanLoader();
    defer lib.close();

    vulkan_loader = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse {
        return error.MissingVkGetInstanceProcAddr;
    };

    const vkb = BaseWrapper.load(loadProc);

    const loader_version = try vkb.enumerateInstanceVersion();
    try printVersionLine("Vulkan loader API version", loader_version);

    const instance_handle = try vkb.createInstance(&.{
        .p_application_info = &.{
            .p_application_name = null,
            .application_version = 0,
            .p_engine_name = null,
            .engine_version = 0,
            .api_version = vk.makeApiVersion(0, 1, 0, 0).toU32(),
        },
    }, null);

    var vki = InstanceWrapper.load(instance_handle, vulkan_loader);
    const instance = Instance.init(instance_handle, &vki);
    defer instance.destroyInstance(null);

    const allocator = std.heap.page_allocator;
    const gpus = try instance.enumeratePhysicalDevicesAlloc(allocator);
    defer allocator.free(gpus);

    if (gpus.len == 0) {
        std.debug.print("No Vulkan physical devices found\n", .{});
        return;
    }

    for (gpus, 0..) |gpu, index| {
        const props = instance.getPhysicalDeviceProperties(gpu);
        try printGpuLine(index, std.mem.sliceTo(&props.device_name, 0), props);
    }
}

fn loadProc(instance: vk.Instance, proc_name: [*:0]const u8) vk.PfnVoidFunction {
    return vulkan_loader(instance, proc_name);
}

fn openVulkanLoader() !std.DynLib {
    const library_names = switch (builtin.os.tag) {
        .linux => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
        .windows => &[_][]const u8{"vulkan-1.dll"},
        .macos => &[_][]const u8{ "libvulkan.1.dylib", "libvulkan.dylib" },
        else => &[_][]const u8{"libvulkan.so.1"},
    };

    for (library_names) |name| {
        return std.DynLib.open(name) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
    }

    return error.VulkanLoaderNotFound;
}

fn printVersionLine(label: []const u8, version: u32) !void {
    const parts = splitVersion(version);

    if (parts.variant == 0) {
        std.debug.print("{s}: {d}.{d}.{d}\n", .{ label, parts.major, parts.minor, parts.patch });
        return;
    }

    std.debug.print("{s}: {d}.{d}.{d}.{d}\n", .{
        label,
        parts.variant,
        parts.major,
        parts.minor,
        parts.patch,
    });
}

fn printGpuLine(index: usize, name: []const u8, props: vk.PhysicalDeviceProperties) !void {
    const parts = splitVersion(props.api_version);

    if (parts.variant == 0) {
        std.debug.print(
            "GPU {d}: {s}, API {d}.{d}.{d}, driver={d}, vendor={d}, device={d}\n",
            .{
                index,
                name,
                parts.major,
                parts.minor,
                parts.patch,
                props.driver_version,
                props.vendor_id,
                props.device_id,
            },
        );
        return;
    }

    std.debug.print(
        "GPU {d}: {s}, API {d}.{d}.{d}.{d}, driver={d}, vendor={d}, device={d}\n",
        .{
            index,
            name,
            parts.variant,
            parts.major,
            parts.minor,
            parts.patch,
            props.driver_version,
            props.vendor_id,
            props.device_id,
        },
    );
}

fn splitVersion(version: u32) VersionParts {
    return .{
        .variant = version >> 29,
        .major = (version >> 22) & 0x7f,
        .minor = (version >> 12) & 0x3ff,
        .patch = version & 0xfff,
    };
}

const VersionParts = struct {
    variant: u32,
    major: u32,
    minor: u32,
    patch: u32,
};
