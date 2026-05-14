APP = 17_ray7_denoiser

.PHONY: build run clean

build:
	zig build

build-release:
	zig build -Doptimize=ReleaseSafe

run:
	zig build run

clean:
	rm -rf zig-out .zig-cache
