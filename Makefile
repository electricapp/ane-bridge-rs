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
CFLAGS        ?= -O2 -Wall -Wextra -Wno-unused-parameter -fno-objc-arc \
                 -fobjc-link-runtime -I$(INCLUDE_DIR)
LDFLAGS_LIB   := -dynamiclib -install_name @rpath/libane_bridge.dylib \
                 -framework Foundation -framework IOSurface -ldl
LDFLAGS_BIN   := -Wl,-rpath,@executable_path/../lib -L$(LIB_DIR) -lane_bridge \
                 -framework Foundation -framework IOSurface

SRCS          := $(SRC_DIR)/ane_private.m $(SRC_DIR)/ane_bridge.m
DYLIB         := $(LIB_DIR)/libane_bridge.dylib

EXAMPLES      := $(BIN_DIR)/identity $(BIN_DIR)/zero_copy

.PHONY: all examples test clean rust rust-test inspect probe-shapes

all: $(DYLIB)

$(DYLIB): $(SRCS) $(INCLUDE_DIR)/ane_bridge.h $(SRC_DIR)/ane_private.h
	@mkdir -p $(LIB_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS_LIB) -o $@ $(SRCS)

examples: $(EXAMPLES)

$(BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS_BIN)

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
