APP = 07_model_textured

.PHONY: build build-release run

build:
	zig build
	cp zig-out/bin/$(APP) ./$(APP)

build-release:
	zig build -Doptimize=ReleaseSafe
	cp zig-out/bin/$(APP) ./$(APP)

run:
	zig build run
