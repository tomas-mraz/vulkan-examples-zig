APP = 06_model

.PHONY: build build-release run

build:
	zig build
	cp zig-out/bin/$(APP) ./$(APP)

build-release:
	zig build -Doptimize=ReleaseSafe
	cp zig-out/bin/$(APP) ./$(APP)

run:
	zig build run
