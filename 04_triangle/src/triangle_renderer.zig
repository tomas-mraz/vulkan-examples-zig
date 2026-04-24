const ash = @import("ash");
const std = @import("std");
const glfw = ash.glfw;
const vk = ash.vk;

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;

const Vertex = extern struct {
    pos: [3]f32,

    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_descriptions = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
    };
};

const triangle_vertices = blk: {
    const radius = 0.065;
    break :blk [_]Vertex{
        .{ .pos = .{ 0.0, -radius, 0.0 } },
        .{ .pos = .{
            radius * @sin(2.0 * std.math.pi / 3.0),
            -radius * @cos(2.0 * std.math.pi / 3.0),
            0.0,
        } },
        .{ .pos = .{
            radius * @sin(4.0 * std.math.pi / 3.0),
            -radius * @cos(4.0 * std.math.pi / 3.0),
            0.0,
        } },
    };
};

const frame_time_warning_threshold_s: f64 = 0.03;
const fps_report_interval_s: f64 = 10.0;
const axis_release_grace_s: f64 = 0.04;
const fps_logging_enabled = false;
const keyboard_event_logging_enabled = false;

const TrianglePushConstants = extern struct {
    angle: f32,
    _padding: f32 = 0,
    offset: [2]f32,
};

const AxisState = struct {
    negative_down: bool = false,
    positive_down: bool = false,
    last_nonzero: i8 = 0,
    release_grace_until: f64 = 0,

    fn press(self: *AxisState, dir: i8) void {
        if (dir < 0) {
            self.negative_down = true;
        } else {
            self.positive_down = true;
        }
        self.last_nonzero = dir;
        self.release_grace_until = 0;
    }

    fn release(self: *AxisState, dir: i8, now: f64) void {
        if (dir < 0) {
            self.negative_down = false;
        } else {
            self.positive_down = false;
        }

        if (!self.negative_down and !self.positive_down) {
            self.last_nonzero = dir;
            self.release_grace_until = now + axis_release_grace_s;
        }
    }

    fn direction(self: *const AxisState, now: f64) i8 {
        if (self.negative_down and self.positive_down) {
            return self.last_nonzero;
        }
        if (self.negative_down) {
            return -1;
        }
        if (self.positive_down) {
            return 1;
        }
        if (now < self.release_grace_until) {
            return self.last_nonzero;
        }
        return 0;
    }
};

const InputState = struct {
    horizontal: AxisState = .{},
    vertical: AxisState = .{},
};

const BufferResource = struct {
    buffer: vk.Buffer = .null_handle,
    memory: vk.DeviceMemory = .null_handle,

    fn deinit(self: *BufferResource, device: vk.DeviceProxy) void {
        if (self.buffer != .null_handle) {
            device.destroyBuffer(self.buffer, null);
            self.buffer = .null_handle;
        }
        if (self.memory != .null_handle) {
            device.freeMemory(self.memory, null);
            self.memory = .null_handle;
        }
    }
};

pub const TriangleRenderer = struct {
    allocator: std.mem.Allocator,
    host: *ash.DesktopHost,
    device: ?vk.DeviceProxy = null,

    vertex_buffer: BufferResource = .{},
    render_pass: vk.RenderPass = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    pipeline: vk.Pipeline = .null_handle,
    framebuffers: []vk.Framebuffer = &.{},

    start_time: f64 = 0,
    last_frame_time: f64 = 0,
    fps_window_start_time: f64 = 0,
    fps_window_frames: u32 = 0,
    triangle_offset: [2]f32 = .{ 0, 0 },
    input: InputState = .{},

    once_built: bool = false,
    sized_built: bool = false,

    pub fn init(allocator: std.mem.Allocator, host: *ash.DesktopHost) TriangleRenderer {
        return .{
            .allocator = allocator,
            .host = host,
        };
    }

    pub fn createOnce(self: *TriangleRenderer, session: *ash.Session) !void {
        const manager = &session.manager.?;
        const device = manager.device orelse return error.DeviceNotInitialized;
        self.device = device;

        self.vertex_buffer = try createVertexBuffer(manager);
        self.start_time = glfw.getTime();
        self.last_frame_time = self.start_time;
        self.fps_window_start_time = self.start_time;
        self.fps_window_frames = 0;
        self.triangle_offset = .{ 0, 0 };
        self.input = .{};
        if (self.host.window) |window| {
            active_renderer = self;
            window.setKeyCallback(keyCallback);
        }
        self.once_built = true;
    }

    pub fn destroyOnce(self: *TriangleRenderer) void {
        if (!self.once_built) {
            return;
        }
        const device = self.device orelse return;
        if (self.host.window) |window| {
            window.setKeyCallback(null);
        }
        if (active_renderer == self) {
            active_renderer = null;
        }
        self.vertex_buffer.deinit(device);
        self.device = null;
        self.once_built = false;
    }

    pub fn createSized(self: *TriangleRenderer, session: *ash.Session, extent: vk.Extent2D) !void {
        _ = extent;
        const device = session.manager.?.device orelse return error.DeviceNotInitialized;

        self.render_pass = try createRenderPass(device, session.swapchain.?);
        errdefer device.destroyRenderPass(self.render_pass, null);

        self.pipeline_layout = try device.createPipelineLayout(&.{
            .set_layout_count = 0,
            .p_set_layouts = undefined,
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&[_]vk.PushConstantRange{.{
                .stage_flags = .{ .vertex_bit = true },
                .offset = 0,
                .size = @sizeOf(TrianglePushConstants),
            }}),
        }, null);
        errdefer device.destroyPipelineLayout(self.pipeline_layout, null);

        self.pipeline = try createPipeline(device, self.pipeline_layout, self.render_pass);
        errdefer device.destroyPipeline(self.pipeline, null);

        self.framebuffers = try createFramebuffers(
            self.allocator,
            device,
            session.swapchain.?,
            self.render_pass,
        );
        errdefer destroyFramebuffers(self.allocator, device, self.framebuffers);

        self.sized_built = true;
    }

    pub fn destroySized(self: *TriangleRenderer) void {
        if (!self.sized_built) {
            return;
        }
        const device = self.device orelse return;
        destroyFramebuffers(self.allocator, device, self.framebuffers);
        self.framebuffers = &.{};
        if (self.pipeline != .null_handle) {
            device.destroyPipeline(self.pipeline, null);
            self.pipeline = .null_handle;
        }
        if (self.pipeline_layout != .null_handle) {
            device.destroyPipelineLayout(self.pipeline_layout, null);
            self.pipeline_layout = .null_handle;
        }
        if (self.render_pass != .null_handle) {
            device.destroyRenderPass(self.render_pass, null);
            self.render_pass = .null_handle;
        }
        self.sized_built = false;
    }

    pub fn draw(self: *TriangleRenderer, session: *ash.Session, frame: *const ash.Frame) !void {
        const device = session.manager.?.device orelse return error.DeviceNotInitialized;

        const now = glfw.getTime();
        const delta_time: f32 = @floatCast(now - self.last_frame_time);
        self.last_frame_time = now;
        self.fps_window_frames += 1;

        if (delta_time > frame_time_warning_threshold_s) {
            std.log.warn(
                "triangle frame hitch: dt={d:.2} ms offset=({d:.3}, {d:.3})",
                .{ delta_time * 1000.0, self.triangle_offset[0], self.triangle_offset[1] },
            );
        }

        const fps_window_elapsed = now - self.fps_window_start_time;
        if (fps_window_elapsed >= fps_report_interval_s) {
            const avg_fps = @as(f64, @floatFromInt(self.fps_window_frames)) / fps_window_elapsed;
            if (fps_logging_enabled) {
                std.log.info(
                    "triangle avg fps: {d:.2} over {d:.2} s ({d} frames)",
                    .{ avg_fps, fps_window_elapsed, self.fps_window_frames },
                );
            }
            self.fps_window_start_time = now;
            self.fps_window_frames = 0;
        }

        const capped_dt = @min(delta_time, 1.0 / 60.0);
        updateTriangleOffset(&self.input, now, capped_dt, &self.triangle_offset);

        const push_constants = TrianglePushConstants{
            .angle = @floatCast((now - self.start_time) * 2.0),
            .offset = self.triangle_offset,
        };

        const clear_value = vk.ClearValue{
            .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 1.0 } },
        };
        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(frame.extent.width),
            .height = @floatFromInt(frame.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };
        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = frame.extent,
        };

        device.cmdBeginRenderPass(frame.cmd, &.{
            .render_pass = self.render_pass,
            .framebuffer = self.framebuffers[@intCast(frame.image_index)],
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = frame.extent,
            },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear_value),
        }, .@"inline");

        device.cmdBindPipeline(frame.cmd, .graphics, self.pipeline);
        device.cmdSetViewport(frame.cmd, 0, &.{viewport});
        device.cmdSetScissor(frame.cmd, 0, &.{scissor});
        device.cmdPushConstants(
            frame.cmd,
            self.pipeline_layout,
            .{ .vertex_bit = true },
            0,
            @sizeOf(TrianglePushConstants),
            &push_constants,
        );

        const offsets = [_]vk.DeviceSize{0};
        device.cmdBindVertexBuffers(frame.cmd, 0, &.{self.vertex_buffer.buffer}, &offsets);
        device.cmdDraw(frame.cmd, triangle_vertices.len, 1, 0, 0);
        device.cmdEndRenderPass(frame.cmd);
    }
};

var active_renderer: ?*TriangleRenderer = null;

fn keyCallback(window: glfw.Window, key: glfw.Key, scancode: i32, action: glfw.Action, mods: glfw.Mods) void {
    _ = window;
    _ = scancode;
    _ = mods;

    const renderer = active_renderer orelse return;
    const timestamp = glfw.getTime();

    if (keyboard_event_logging_enabled) {
        std.log.info("triangle key event: t={d:.3}s key={} action={}", .{ timestamp, key, action });
    }

    switch (key) {
        .a => switch (action) {
            .press => renderer.input.horizontal.press(-1),
            .release => renderer.input.horizontal.release(-1, timestamp),
            .repeat => {},
        },
        .d => switch (action) {
            .press => renderer.input.horizontal.press(1),
            .release => renderer.input.horizontal.release(1, timestamp),
            .repeat => {},
        },
        .w => switch (action) {
            .press => renderer.input.vertical.press(-1),
            .release => renderer.input.vertical.release(-1, timestamp),
            .repeat => {},
        },
        .s => switch (action) {
            .press => renderer.input.vertical.press(1),
            .release => renderer.input.vertical.release(1, timestamp),
            .repeat => {},
        },
        else => {},
    }

    if (keyboard_event_logging_enabled and (key == .a or key == .d or key == .w or key == .s)) {
        std.log.info(
            "triangle input state: t={d:.3}s h=({},{}) v=({},{}) active=({d},{d})",
            .{
                timestamp,
                renderer.input.horizontal.negative_down,
                renderer.input.horizontal.positive_down,
                renderer.input.vertical.negative_down,
                renderer.input.vertical.positive_down,
                renderer.input.horizontal.direction(timestamp),
                renderer.input.vertical.direction(timestamp),
            },
        );
    }
}

fn createVertexBuffer(manager: *const ash.Manager) !BufferResource {
    const device = manager.device orelse return error.DeviceNotInitialized;

    var result = BufferResource{};
    errdefer result.deinit(device);

    result.buffer = try device.createBuffer(&.{
        .size = @sizeOf(@TypeOf(triangle_vertices)),
        .usage = .{ .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);

    const requirements = device.getBufferMemoryRequirements(result.buffer);
    result.memory = try manager.allocate(requirements, .{
        .host_visible_bit = true,
        .host_coherent_bit = true,
    });
    try device.bindBufferMemory(result.buffer, result.memory, 0);

    const mapped = try device.mapMemory(result.memory, 0, vk.WHOLE_SIZE, .{});
    defer device.unmapMemory(result.memory);

    const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(mapped));
    @memcpy(gpu_vertices, triangle_vertices[0..]);

    return result;
}

fn createRenderPass(device: vk.DeviceProxy, swapchain: ash.Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    return try device.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
    }, null);
}

fn createPipeline(device: vk.DeviceProxy, pipeline_layout: vk.PipelineLayout, render_pass: vk.RenderPass) !vk.Pipeline {
    const vert = try device.createShaderModule(&.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(&vert_spv),
    }, null);
    defer device.destroyShaderModule(vert, null);

    const frag = try device.createShaderModule(&.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(&frag_spv),
    }, null);
    defer device.destroyShaderModule(frag, null);

    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex_bit = true }, .module = vert, .p_name = "main" },
        .{ .stage = .{ .fragment_bit = true }, .module = frag, .p_name = "main" },
    };

    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_descriptions.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_descriptions,
    };
    const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .topology = .triangle_list,
        .primitive_restart_enable = .false,
    };
    const viewport_state = vk.PipelineViewportStateCreateInfo{
        .viewport_count = 1,
        .p_viewports = undefined,
        .scissor_count = 1,
        .p_scissors = undefined,
    };
    const rasterization = vk.PipelineRasterizationStateCreateInfo{
        .depth_clamp_enable = .false,
        .rasterizer_discard_enable = .false,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = .false,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };
    const multisample = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = .false,
        .min_sample_shading = 1,
        .alpha_to_coverage_enable = .false,
        .alpha_to_one_enable = .false,
    };
    const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .false,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };
    const color_blend = vk.PipelineColorBlendStateCreateInfo{
        .logic_op_enable = .false,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_blend_attachment),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };
    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const dynamic_state = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = dynamic_states.len,
        .p_dynamic_states = &dynamic_states,
    };

    const create_info = vk.GraphicsPipelineCreateInfo{
        .stage_count = shader_stages.len,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &input_assembly,
        .p_tessellation_state = null,
        .p_viewport_state = &viewport_state,
        .p_rasterization_state = &rasterization,
        .p_multisample_state = &multisample,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &color_blend,
        .p_dynamic_state = &dynamic_state,
        .layout = pipeline_layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try device.createGraphicsPipelines(.null_handle, &.{create_info}, null, (&pipeline)[0..1]);
    return pipeline;
}

fn createFramebuffers(
    allocator: std.mem.Allocator,
    device: vk.DeviceProxy,
    swapchain: ash.Swapchain,
    render_pass: vk.RenderPass,
) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var created: usize = 0;
    errdefer {
        for (framebuffers[0..created]) |framebuffer| {
            device.destroyFramebuffer(framebuffer, null);
        }
    }

    for (swapchain.swap_images, 0..) |swap_image, index| {
        framebuffers[index] = try device.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swap_image.view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        created += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(allocator: std.mem.Allocator, device: vk.DeviceProxy, framebuffers: []vk.Framebuffer) void {
    for (framebuffers) |framebuffer| {
        device.destroyFramebuffer(framebuffer, null);
    }
    allocator.free(framebuffers);
}

fn updateTriangleOffset(input: *const InputState, now: f64, delta_time: f32, offset: *[2]f32) void {
    const move_speed: f32 = 0.625;
    const step = move_speed * delta_time;
    const max_offset: f32 = 0.8;

    offset[0] += @as(f32, @floatFromInt(input.horizontal.direction(now))) * step;
    offset[1] += @as(f32, @floatFromInt(input.vertical.direction(now))) * step;

    offset[0] = std.math.clamp(offset[0], -max_offset, max_offset);
    offset[1] = std.math.clamp(offset[1], -max_offset, max_offset);
}
