# ane-bridge — top-level Makefile.
#
# Targets:
#   make            — build libane_bridge.dylib
#   make examples   — build C examples (depends on lib)
#   make test       — build + run the C identity example
#   make rust       — cargo build the Rust workspace (rebuilds C via cc crate)
#   make rust-test  — cargo run the Rust identity example
#   make clean      — remove build/

PREFIX        ?= build
LIB_DIR       := $(PREFIX)/lib
BIN_DIR       := $(PREFIX)/bin
INCLUDE_DIR   := c/include
SRC_DIR       := c/src
EXAMPLES_DIR  := c/examples

CC            ?= xcrun clang

# Maximum-strictness warning policy for the C/Obj-C side.
#
# `-Weverything -Werror` opts into every warning clang knows and makes them
# fatal. The `-Wno-*` list opts out ONLY of categories that fight idioms this
# codebase deliberately uses; each one is load-bearing — do not drop a line
# without checking what it re-enables:
#   declaration-after-statement  C99 mixed declarations are intentional
#   objc-messaging-id            the private _ANE* framework is dispatched via `id`
#   cast-function-type-strict    required typed-cast idiom for objc_msgSend
#   padded                       struct/inter-field padding is not actionable
#   poison-system-directories    cross-compile-only noise; irrelevant on-host
#   switch-default               exhaustive switches use a trailing fallback return
#                                (adding `default:` would trip -Wcovered-switch-default)
#   unused-parameter             callback / function-pointer signatures carry unused args
#   double-promotion             benign float->double in numeric / printf example code
WARN          := -Weverything -Werror \
                 -Wno-declaration-after-statement \
                 -Wno-objc-messaging-id \
                 -Wno-cast-function-type-strict \
                 -Wno-padded \
                 -Wno-poison-system-directories \
                 -Wno-switch-default \
                 -Wno-unused-parameter \
                 -Wno-double-promotion

CFLAGS        ?= -O2 $(WARN) -fno-objc-arc -fobjc-link-runtime -I$(INCLUDE_DIR)
LDFLAGS_LIB   := -dynamiclib -install_name @rpath/libane_bridge.dylib \
                 -framework Foundation -framework IOSurface -ldl
LDFLAGS_BIN   := -Wl,-rpath,@executable_path/../lib -L$(LIB_DIR) -lane_bridge \
                 -framework Foundation -framework IOSurface

SRCS          := $(SRC_DIR)/ane_private.m $(SRC_DIR)/ane_bridge.m
DYLIB         := $(LIB_DIR)/libane_bridge.dylib

EXAMPLES      := $(BIN_DIR)/identity $(BIN_DIR)/zero_copy $(BIN_DIR)/gpu_to_ane $(BIN_DIR)/chain_identity $(BIN_DIR)/chain_file $(BIN_DIR)/ane_vs_sme

.PHONY: all examples test clean rust rust-test inspect probe-shapes

all: $(DYLIB)

$(DYLIB): $(SRCS) $(INCLUDE_DIR)/ane_bridge.h $(SRC_DIR)/ane_private.h
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_LIB) -o $@ $(SRCS)

examples: $(EXAMPLES)

$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_BIN)

$(BIN_DIR)/gpu_to_ane: $(EXAMPLES_DIR)/gpu_to_ane.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -fobjc-arc -framework Metal -o $@ $< $(LDFLAGS_BIN)

$(BIN_DIR)/ane_vs_sme: $(EXAMPLES_DIR)/ane_vs_sme.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -fobjc-arc \
	    -DACCELERATE_NEW_LAPACK -framework Accelerate \
	    -o $@ $< $(LDFLAGS_BIN)

test: examples
	@uv run --project .. python tools/make_identity_model.py $(PREFIX)/identity || \
	    python3 tools/make_identity_model.py $(PREFIX)/identity
	./$(BIN_DIR)/identity $(PREFIX)/identity/model.mil $(PREFIX)/identity/weights.bin

rust:
	cd rust && cargo build --workspace

rust-test:
	@uv run --project .. python tools/make_identity_model.py $(PREFIX)/identity || \
	    python3 tools/make_identity_model.py $(PREFIX)/identity
	cd rust && cargo run --example identity -- \
	    ../$(PREFIX)/identity/model.mil ../$(PREFIX)/identity/weights.bin

# One-off discovery tool: enumerates methods/properties on the private
# AppleNeuralEngine classes so we can find shape-introspection APIs.
inspect:
	@mkdir -p $(BIN_DIR)
	$(CC) -O0 -fno-objc-arc -o $(BIN_DIR)/inspect tools/inspect_classes.m \
	    -framework Foundation -ldl
	./$(BIN_DIR)/inspect

# Open a real model and call every promising shape-accessor we found
# during `make inspect`. Prints the returned object structures so we
# can decide which ones expose input/output shapes.
probe-shapes: $(DYLIB)
	@mkdir -p $(BIN_DIR) $(PREFIX)/identity
	@uv run --project .. python tools/make_identity_model.py $(PREFIX)/identity || \
	    python3 tools/make_identity_model.py $(PREFIX)/identity
	$(CC) -O0 -fno-objc-arc -o $(BIN_DIR)/probe_shapes tools/probe_shapes.m \
	    -framework Foundation -framework IOSurface -ldl
	./$(BIN_DIR)/probe_shapes $(PREFIX)/identity/model.mil $(PREFIX)/identity/weights.bin

clean:
	rm -rf $(PREFIX) rust/target
