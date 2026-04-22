APP = 03_draw
ZIG ?= /home/tomas/Applications/zig-x86_64-linux-0.17.0-dev.56+a8226cd53/zig

.PHONY: build build-release run

build:
	$(ZIG) build
	cp zig-out/bin/$(APP) ./$(APP)

build-release:
	$(ZIG) build -Doptimize=ReleaseSafe
	cp zig-out/bin/$(APP) ./$(APP)

run:
	$(ZIG) build run
