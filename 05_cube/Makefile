APP = 05_cube

ANDROID_TARGET ?= aarch64-linux-android
ANDROID_ABI ?= arm64-v8a
ANDROID_DIR = android
ANDROID_JNILIBS = $(ANDROID_DIR)/app/src/main/jniLibs/$(ANDROID_ABI)
ANDROID_APK_DEBUG = $(ANDROID_DIR)/app/build/outputs/apk/debug/app-$(ANDROID_ABI)-debug.apk
ANDROID_APK_RELEASE = $(ANDROID_DIR)/app/build/outputs/apk/release/app-$(ANDROID_ABI)-release-unsigned.apk

.PHONY: build build-release run build-android build-android-release clean-android

build:
	zig build
	mv zig-out/bin/$(APP) ./$(APP)

build-release:
	zig build -Doptimize=ReleaseSafe
	mv zig-out/bin/$(APP) ./$(APP)

run:
	zig build run

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
