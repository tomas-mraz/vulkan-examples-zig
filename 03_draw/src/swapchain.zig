const std = @import("std");
const vk = @import("vulkan");

const Allocator = std.mem.Allocator;
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;

pub const Swapchain = struct {
    pub const PresentState = enum {
        optimal,
        suboptimal,
    };

    gc: *const GraphicsContext,
    allocator: Allocator,

    surface_format: vk.SurfaceFormatKHR,
    extent: vk.Extent2D,
    handle: vk.SwapchainKHR,

    swap_images: []SwapImage,
    image_index: u32,
    next_image_acquired: vk.Semaphore,

    pub fn init(gc: *const GraphicsContext, allocator: Allocator, extent: vk.Extent2D) !Swapchain {
        return try initRecycle(gc, allocator, extent, .null_handle);
    }

    pub fn deinit(self: Swapchain) void {
        if (self.handle == .null_handle) return;
        self.deinitExceptSwapchain();
        self.gc.dev.destroySwapchainKHR(self.handle, null);
    }

    pub fn recreate(self: *Swapchain, new_extent: vk.Extent2D) !void {
        const gc = self.gc;
        const allocator = self.allocator;
        const old_handle = self.handle;

        try gc.dev.queueWaitIdle(gc.present_queue.handle);
        self.deinitExceptSwapchain();

        self.handle = .null_handle;
        self.* = initRecycle(gc, allocator, new_extent, old_handle) catch |err| switch (err) {
            error.SwapchainCreationFailed => {
                gc.dev.destroySwapchainKHR(old_handle, null);
                return err;
            },
            else => return err,
        };
    }

    pub fn waitForAllFences(self: Swapchain) !void {
        for (self.swap_images) |swap_image| {
            try swap_image.waitForFence(self.gc);
        }
    }

    pub fn present(self: *Swapchain, cmdbuf: vk.CommandBuffer) !PresentState {
        const current = &self.swap_images[self.image_index];
        try current.waitForFence(self.gc);
        try self.gc.dev.resetFences(&.{current.frame_fence});

        const wait_stage = [_]vk.PipelineStageFlags{.{ .top_of_pipe_bit = true }};
        try self.gc.dev.queueSubmit(self.gc.graphics_queue.handle, &.{.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&current.image_acquired),
            .p_wait_dst_stage_mask = &wait_stage,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmdbuf),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&current.render_finished),
        }}, current.frame_fence);

        _ = try self.gc.dev.queuePresentKHR(self.gc.present_queue.handle, &.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&current.render_finished),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&self.handle),
            .p_image_indices = @ptrCast(&self.image_index),
        });

        const result = try self.gc.dev.acquireNextImageKHR(
            self.handle,
            std.math.maxInt(u64),
            self.next_image_acquired,
            .null_handle,
        );

        std.mem.swap(vk.Semaphore, &self.swap_images[result.image_index].image_acquired, &self.next_image_acquired);
        self.image_index = result.image_index;

        return switch (result.result) {
            .success => .optimal,
            .suboptimal_khr => .suboptimal,
            else => unreachable,
        };
    }

    fn initRecycle(gc: *const GraphicsContext, allocator: Allocator, extent: vk.Extent2D, old_handle: vk.SwapchainKHR) !Swapchain {
        const caps = try gc.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(gc.pdev, gc.surface);
        const actual_extent = findActualExtent(caps, extent);
        if (actual_extent.width == 0 or actual_extent.height == 0) {
            return error.InvalidSurfaceDimensions;
        }

        const surface_format = try findSurfaceFormat(gc, allocator);
        const present_mode = try findPresentMode(gc, allocator);

        var image_count = caps.min_image_count + 1;
        if (caps.max_image_count > 0) {
            image_count = @min(image_count, caps.max_image_count);
        }

        const qfi = [_]u32{ gc.graphics_queue.family, gc.present_queue.family };
        const sharing_mode: vk.SharingMode = if (gc.graphics_queue.family != gc.present_queue.family) .concurrent else .exclusive;

        const handle = gc.dev.createSwapchainKHR(&.{
            .surface = gc.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = actual_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = sharing_mode,
            .queue_family_index_count = qfi.len,
            .p_queue_family_indices = &qfi,
            .pre_transform = caps.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = .true,
            .old_swapchain = old_handle,
        }, null) catch {
            return error.SwapchainCreationFailed;
        };
        errdefer gc.dev.destroySwapchainKHR(handle, null);

        if (old_handle != .null_handle) {
            gc.dev.destroySwapchainKHR(old_handle, null);
        }

        const swap_images = try initSwapchainImages(gc, handle, surface_format.format, allocator);
        errdefer {
            for (swap_images) |swap_image| swap_image.deinit(gc);
            allocator.free(swap_images);
        }

        var next_image_acquired = try gc.dev.createSemaphore(&.{}, null);
        errdefer gc.dev.destroySemaphore(next_image_acquired, null);

        const result = try gc.dev.acquireNextImageKHR(handle, std.math.maxInt(u64), next_image_acquired, .null_handle);
        if (result.result == .not_ready or result.result == .timeout) {
            return error.ImageAcquireFailed;
        }

        std.mem.swap(vk.Semaphore, &swap_images[result.image_index].image_acquired, &next_image_acquired);
        return .{
            .gc = gc,
            .allocator = allocator,
            .surface_format = surface_format,
            .extent = actual_extent,
            .handle = handle,
            .swap_images = swap_images,
            .image_index = result.image_index,
            .next_image_acquired = next_image_acquired,
        };
    }

    fn deinitExceptSwapchain(self: Swapchain) void {
        for (self.swap_images) |swap_image| swap_image.deinit(self.gc);
        self.allocator.free(self.swap_images);
        self.gc.dev.destroySemaphore(self.next_image_acquired, null);
    }
};

const SwapImage = struct {
    image: vk.Image,
    view: vk.ImageView,
    image_acquired: vk.Semaphore,
    render_finished: vk.Semaphore,
    frame_fence: vk.Fence,

    fn init(gc: *const GraphicsContext, image: vk.Image, format: vk.Format) !SwapImage {
        const view = try gc.dev.createImageView(&.{
            .image = image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer gc.dev.destroyImageView(view, null);

        const image_acquired = try gc.dev.createSemaphore(&.{}, null);
        errdefer gc.dev.destroySemaphore(image_acquired, null);

        const render_finished = try gc.dev.createSemaphore(&.{}, null);
        errdefer gc.dev.destroySemaphore(render_finished, null);

        const frame_fence = try gc.dev.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
        errdefer gc.dev.destroyFence(frame_fence, null);

        return .{
            .image = image,
            .view = view,
            .image_acquired = image_acquired,
            .render_finished = render_finished,
            .frame_fence = frame_fence,
        };
    }

    fn deinit(self: SwapImage, gc: *const GraphicsContext) void {
        self.waitForFence(gc) catch return;
        gc.dev.destroyImageView(self.view, null);
        gc.dev.destroySemaphore(self.image_acquired, null);
        gc.dev.destroySemaphore(self.render_finished, null);
        gc.dev.destroyFence(self.frame_fence, null);
    }

    fn waitForFence(self: SwapImage, gc: *const GraphicsContext) !void {
        _ = try gc.dev.waitForFences(&.{self.frame_fence}, .true, std.math.maxInt(u64));
    }
};

fn initSwapchainImages(gc: *const GraphicsContext, swapchain: vk.SwapchainKHR, format: vk.Format, allocator: Allocator) ![]SwapImage {
    const images = try gc.dev.getSwapchainImagesAllocKHR(swapchain, allocator);
    defer allocator.free(images);

    const swap_images = try allocator.alloc(SwapImage, images.len);
    errdefer allocator.free(swap_images);

    var index: usize = 0;
    errdefer for (swap_images[0..index]) |swap_image| swap_image.deinit(gc);

    for (images) |image| {
        swap_images[index] = try SwapImage.init(gc, image, format);
        index += 1;
    }

    return swap_images;
}

fn findSurfaceFormat(gc: *const GraphicsContext, allocator: Allocator) !vk.SurfaceFormatKHR {
    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };

    const surface_formats = try gc.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(gc.pdev, gc.surface, allocator);
    defer allocator.free(surface_formats);

    for (surface_formats) |surface_format| {
        if (std.meta.eql(surface_format, preferred)) {
            return preferred;
        }
    }

    return surface_formats[0];
}

fn findPresentMode(gc: *const GraphicsContext, allocator: Allocator) !vk.PresentModeKHR {
    const present_modes = try gc.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(gc.pdev, gc.surface, allocator);
    defer allocator.free(present_modes);

    const preferred = [_]vk.PresentModeKHR{ .mailbox_khr, .immediate_khr };
    for (preferred) |mode| {
        if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
            return mode;
        }
    }

    return .fifo_khr;
}

fn findActualExtent(caps: vk.SurfaceCapabilitiesKHR, extent: vk.Extent2D) vk.Extent2D {
    if (caps.current_extent.width != 0xFFFF_FFFF) {
        return caps.current_extent;
    }

    return .{
        .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
        .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
    };
}
