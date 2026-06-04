# Probes and one-off discovery tools that link the bridge dylib.
# Build everything with `make -C tools all`. Each binary lands in
# ../build/bin alongside the C examples.

PREFIX        := ../build
LIB_DIR       := $(PREFIX)/lib
BIN_DIR       := $(PREFIX)/bin
INC_DIR       := ../c/include

CC            ?= xcrun clang
WARN          := -Wall -Wextra -Wno-unused-parameter
CFLAGS        := -O2 $(WARN) -fno-objc-arc -I$(INC_DIR)
LDFLAGS       := -Wl,-rpath,@executable_path/../lib -L$(LIB_DIR) -lane_bridge \
                 -framework Foundation -framework IOSurface -ldl

DYLIB         := $(LIB_DIR)/libane_bridge.dylib

BINS := \
    $(BIN_DIR)/explore_unknowns \
    $(BIN_DIR)/inspect_classes \
    $(BIN_DIR)/probe_shapes \
    $(BIN_DIR)/probe_shared_events \
    $(BIN_DIR)/probe_chain \
    $(BIN_DIR)/probe_chain_entitlements \
    $(BIN_DIR)/probe_perfstats \
    $(BIN_DIR)/probe_lifecycle \
    $(BIN_DIR)/power_witness

.PHONY: all clean

all: $(BINS)

# Discovery / inspection probes — no bridge dylib needed.
$(BIN_DIR)/explore_unknowns: explore_unknowns.m
	@mkdir -p $(BIN_DIR)
	$(CC) -O0 -fno-objc-arc -framework Foundation -ldl -o $@ $<

$(BIN_DIR)/inspect_classes: inspect_classes.m
	@mkdir -p $(BIN_DIR)
	$(CC) -O0 -fno-objc-arc -framework Foundation -ldl -o $@ $<

# Bridge-linked probes.
$(BIN_DIR)/probe_shapes: probe_shapes.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -framework IOSurface -o $@ $< $(LDFLAGS)

$(BIN_DIR)/probe_shared_events: probe_shared_events.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fobjc-arc $(WARN) -I$(INC_DIR) \
	    -framework IOSurface -framework Metal \
	    -o $@ $< $(LDFLAGS)

$(BIN_DIR)/probe_chain: probe_chain.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fobjc-arc $(WARN) -I$(INC_DIR) \
	    -framework IOSurface -framework Metal \
	    -o $@ $< $(LDFLAGS)

$(BIN_DIR)/probe_chain_entitlements: probe_chain_entitlements.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fno-objc-arc $(WARN) -I$(INC_DIR) \
	    -framework IOSurface -framework Metal \
	    -o $@ $< $(LDFLAGS)

$(BIN_DIR)/probe_perfstats: probe_perfstats.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fobjc-arc $(WARN) -I$(INC_DIR) \
	    -framework IOSurface \
	    -o $@ $< $(LDFLAGS)

$(BIN_DIR)/probe_lifecycle: probe_lifecycle.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fobjc-arc $(WARN) -I$(INC_DIR) \
	    -framework IOSurface -lz \
	    -o $@ $< $(LDFLAGS)

$(BIN_DIR)/power_witness: power_witness.m $(DYLIB)
	@mkdir -p $(BIN_DIR)
	$(CC) -O2 -fobjc-arc $(WARN) -DACCELERATE_NEW_LAPACK -I$(INC_DIR) \
	    -framework Accelerate -framework IOSurface \
	    -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS)
