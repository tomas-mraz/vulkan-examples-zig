const builtin = @import("builtin");
const std = @import("std");
const glfw = @import("glfw");
const vk = @import("vulkan");

const Allocator = std.mem.Allocator;

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;

pub const Instance = vk.InstanceProxy;
pub const Device = vk.DeviceProxy;

const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

fn getGlfwInstanceProcAddr(instance: vk.Instance, proc_name: [*:0]const u8) vk.PfnVoidFunction {
    return @ptrCast(glfw.getInstanceProcAddress(
        if (instance == .null_handle) null else @ptrFromInt(@intFromEnum(instance)),
        proc_name,
    ));
}

pub const GraphicsContext = struct {
    allocator: Allocator,

    vkb: BaseWrapper,
    instance: Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    dev: Device,
    graphics_queue: Queue,
    present_queue: Queue,

    pub fn init(allocator: Allocator, app_name: [*:0]const u8, window: glfw.Window) !GraphicsContext {
        var self: GraphicsContext = undefined;
        self.allocator = allocator;

        if (!glfw.vulkanSupported()) {
            return error.VulkanNotSupported;
        }

        self.vkb = BaseWrapper.load(getGlfwInstanceProcAddr);

        var extension_names: std.ArrayList([*:0]const u8) = .empty;
        defer extension_names.deinit(allocator);

        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse {
            return error.MissingRequiredInstanceExtensions;
        };
        try extension_names.appendSlice(allocator, glfw_extensions);

        var flags = vk.InstanceCreateFlags{};
        if (builtin.os.tag.isDarwin()) {
            try appendIfMissing(&extension_names, allocator, vk.extensions.khr_portability_enumeration.name);
            try appendIfMissing(&extension_names, allocator, vk.extensions.khr_get_physical_device_properties_2.name);
            flags.enumerate_portability_bit_khr = true;
        }

        const instance_handle = try self.vkb.createInstance(&.{
            .flags = flags,
            .p_application_info = &.{
                .p_application_name = app_name,
                .application_version = vk.makeApiVersion(0, 0, 0, 0).toU32(),
                .p_engine_name = app_name,
                .engine_version = vk.makeApiVersion(0, 0, 0, 0).toU32(),
                .api_version = vk.makeApiVersion(0, 1, 0, 0).toU32(),
            },
            .enabled_extension_count = @intCast(extension_names.items.len),
            .pp_enabled_extension_names = extension_names.items.ptr,
        }, null);

        const vki = try allocator.create(InstanceWrapper);
        errdefer allocator.destroy(vki);
        vki.* = InstanceWrapper.load(instance_handle, self.vkb.dispatch.vkGetInstanceProcAddr.?);
        self.instance = Instance.init(instance_handle, vki);
        errdefer self.instance.destroyInstance(null);

        self.surface = try createSurface(self.instance, window);
        errdefer self.instance.destroySurfaceKHR(self.surface, null);

        const candidate = try pickPhysicalDevice(self.instance, allocator, self.surface);
        self.pdev = candidate.pdev;
        self.props = candidate.props;

        const device_handle = try initializeCandidate(self.instance, candidate);

        const vkd = try allocator.create(DeviceWrapper);
        errdefer allocator.destroy(vkd);
        vkd.* = DeviceWrapper.load(device_handle, self.instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
        self.dev = Device.init(device_handle, vkd);
        errdefer self.dev.destroyDevice(null);

        self.graphics_queue = Queue.init(self.dev, candidate.queues.graphics_family);
        self.present_queue = Queue.init(self.dev, candidate.queues.present_family);
        self.mem_props = self.instance.getPhysicalDeviceMemoryProperties(self.pdev);

        return self;
    }

    pub fn deinit(self: GraphicsContext) void {
        self.dev.destroyDevice(null);
        self.instance.destroySurfaceKHR(self.surface, null);
        self.instance.destroyInstance(null);
        self.allocator.destroy(self.dev.wrapper);
        self.allocator.destroy(self.instance.wrapper);
    }

    pub fn deviceName(self: *const GraphicsContext) []const u8 {
        return std.mem.sliceTo(&self.props.device_name, 0);
    }

    pub fn findMemoryTypeIndex(self: GraphicsContext, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @as(u5, @truncate(i))) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(i);
            }
        }
        return error.NoSuitableMemoryType;
    }

    pub fn allocate(self: GraphicsContext, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.dev.allocateMemory(&.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }
};

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(device: Device, family: u32) Queue {
        return .{
            .handle = device.getDeviceQueue(family, 0),
            .family = family,
        };
    }
};

fn appendIfMissing(list: *std.ArrayList([*:0]const u8), allocator: Allocator, value: [*:0]const u8) !void {
    for (list.items) |item| {
        if (std.mem.eql(u8, std.mem.span(item), std.mem.span(value))) {
            return;
        }
    }
    try list.append(allocator, value);
}

fn createSurface(instance: Instance, window: glfw.Window) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    const result: vk.Result = @enumFromInt(glfw.createWindowSurface(instance.handle, window, null, &surface));
    if (result != .success) {
        return error.SurfaceInitFailed;
    }
    return surface;
}

fn initializeCandidate(instance: Instance, candidate: DeviceCandidate) !vk.Device {
    const priority = [_]f32{1};
    const queue_create_infos = [_]vk.DeviceQueueCreateInfo{
        .{
            .queue_family_index = candidate.queues.graphics_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
        .{
            .queue_family_index = candidate.queues.present_family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        },
    };

    const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.present_family) 1 else 2;

    return try instance.createDevice(candidate.pdev, &.{
        .queue_create_info_count = queue_count,
        .p_queue_create_infos = &queue_create_infos,
        .enabled_extension_count = required_device_extensions.len,
        .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
    }, null);
}

const DeviceCandidate = struct {
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    queues: QueueAllocation,
};

const QueueAllocation = struct {
    graphics_family: u32,
    present_family: u32,
};

fn pickPhysicalDevice(instance: Instance, allocator: Allocator, surface: vk.SurfaceKHR) !DeviceCandidate {
    const pdevs = try instance.enumeratePhysicalDevicesAlloc(allocator);
    defer allocator.free(pdevs);

    for (pdevs) |pdev| {
        if (try checkSuitable(instance, pdev, allocator, surface)) |candidate| {
            return candidate;
        }
    }

    return error.NoSuitableDevice;
}

fn checkSuitable(
    instance: Instance,
    pdev: vk.PhysicalDevice,
    allocator: Allocator,
    surface: vk.SurfaceKHR,
) !?DeviceCandidate {
    if (!try checkExtensionSupport(instance, pdev, allocator)) {
        return null;
    }

    if (!try checkSurfaceSupport(instance, pdev, surface)) {
        return null;
    }

    if (try allocateQueues(instance, pdev, allocator, surface)) |allocation| {
        return DeviceCandidate{
            .pdev = pdev,
            .props = instance.getPhysicalDeviceProperties(pdev),
            .queues = allocation,
        };
    }

    return null;
}

fn allocateQueues(instance: Instance, pdev: vk.PhysicalDevice, allocator: Allocator, surface: vk.SurfaceKHR) !?QueueAllocation {
    const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, allocator);
    defer allocator.free(families);

    var graphics_family: ?u32 = null;
    var present_family: ?u32 = null;

    for (families, 0..) |properties, i| {
        const family: u32 = @intCast(i);

        if (graphics_family == null and properties.queue_flags.graphics_bit) {
            graphics_family = family;
        }

        if (present_family == null and (try instance.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == .true) {
            present_family = family;
        }
    }

    if (graphics_family != null and present_family != null) {
        return .{
            .graphics_family = graphics_family.?,
            .present_family = present_family.?,
        };
    }

    return null;
}

fn checkSurfaceSupport(instance: Instance, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
    var format_count: u32 = undefined;
    _ = try instance.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

    var present_mode_count: u32 = undefined;
    _ = try instance.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

    return format_count > 0 and present_mode_count > 0;
}

fn checkExtensionSupport(instance: Instance, pdev: vk.PhysicalDevice, allocator: Allocator) !bool {
    const props = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, allocator);
    defer allocator.free(props);

    for (required_device_extensions) |required_ext| {
        for (props) |prop| {
            if (std.mem.eql(u8, std.mem.span(required_ext), std.mem.sliceTo(&prop.extension_name, 0))) {
                break;
            }
        } else {
            return false;
        }
    }

    return true;
}
