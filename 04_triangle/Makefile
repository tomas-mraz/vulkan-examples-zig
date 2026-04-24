APP = 04_triangle

.PHONY: build build-release run

build:
	zig build
	mv zig-out/bin/$(APP) ./$(APP)

build-release:
	zig build -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP) ./$(APP)

run:
	zig build run
