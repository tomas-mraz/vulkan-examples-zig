APP = 05_cube

ANDROID_TARGET ?= aarch64-linux-android
ANDROID_ABI ?= arm64-v8a
ANDROID_DIR = android
ANDROID_JNILIBS = $(ANDROID_DIR)/app/src/main/jniLibs/$(ANDROID_ABI)
ANDROID_APK_DEBUG = $(ANDROID_DIR)/app/build/outputs/apk/debug/app-$(ANDROID_ABI)-debug.apk
ANDROID_APK_RELEASE = $(ANDROID_DIR)/app/build/outputs/apk/release/app-$(ANDROID_ABI)-release-unsigned.apk

WINDOWS_TARGET ?= x86_64-windows-gnu
WINDOWS_ARM64_TARGET ?= aarch64-windows-gnu

# macOS cross-compile from Linux needs a staged macOS SDK via SDKROOT;
# native builds on macOS pick the SDK up automatically via xcrun.
MACOS_TARGET ?= aarch64-macos-none

.PHONY: build build-release run clean \
        build-android build-android-release clean-android \
        build-windows build-windows-release build-windows-arm64 build-windows-arm64-release clean-windows \
        build-macos build-macos-release clean-macos

run:
	zig build run

build:
	zig build
	mv zig-out/bin/$(APP) ./$(APP)

build-release:
	zig build -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP) ./$(APP)

clean:
	rm $(APP)

build-android:
	zig build -Dtarget=$(ANDROID_TARGET)
	mkdir -p $(ANDROID_JNILIBS)
	mv zig-out/lib/lib$(APP).so $(ANDROID_JNILIBS)/
	cd $(ANDROID_DIR) && ./gradlew assembleDebug
	mv $(ANDROID_APK_DEBUG) ./$(APP)-arm64-debug.apk

build-android-release:
	zig build -Dtarget=$(ANDROID_TARGET) -Doptimize=ReleaseSafe
	mkdir -p $(ANDROID_JNILIBS)
	mv zig-out/lib/lib$(APP).so $(ANDROID_JNILIBS)/
	cd $(ANDROID_DIR) && ./gradlew assembleRelease
	mv $(ANDROID_APK_RELEASE) ./$(APP)-arm64-release-unsigned.apk

clean-android:
	rm -r $(ANDROID_DIR)/app/build $(ANDROID_DIR)/app/src/main/jniLibs
	rm -f $(APP)-arm64-debug.apk $(APP)-arm64-release-unsigned.apk

build-windows:
	zig build -Dtarget=$(WINDOWS_TARGET)
	mv zig-out/bin/$(APP).exe ./$(APP).exe

build-windows-release:
	zig build -Dtarget=$(WINDOWS_TARGET) -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP).exe ./$(APP).exe

build-windows-arm64:
	zig build -Dtarget=$(WINDOWS_ARM64_TARGET)
	mv zig-out/bin/$(APP).exe ./$(APP)-arm64.exe

build-windows-arm64-release:
	zig build -Dtarget=$(WINDOWS_ARM64_TARGET) -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP).exe ./$(APP)-arm64.exe

clean-windows:
	rm -f $(APP).exe $(APP).pdb $(APP)-arm64.exe $(APP)-arm64.pdb

build-macos:
	zig build -Dtarget=$(MACOS_TARGET)
	mv zig-out/bin/$(APP) ./$(APP)-macos-arm64

build-macos-release:
	zig build -Dtarget=$(MACOS_TARGET) -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP) ./$(APP)-macos-arm64

clean-macos:
	rm -f $(APP)-macos-arm64
