EA ?= $(HOME)/projects/eacompute/target/release/ea
CC ?= gcc
CFLAGS = -O2 -Wall -Isrc
BUILD = build
LIB = $(BUILD)/lib

.PHONY: all kernels test clean

all: kernels test

kernels:
	@./build_kernels.sh

test: kernels
	$(CC) $(CFLAGS) tests/test_i2s.c -L$(LIB) -lbitnet_i2s -o $(BUILD)/test_i2s -lm
	$(CC) $(CFLAGS) tests/test_quant.c -L$(LIB) -lbitnet_quant -o $(BUILD)/test_quant -lm
	$(CC) $(CFLAGS) tests/test_lut.c -L$(LIB) -lbitnet_lut -o $(BUILD)/test_lut -lm
	$(CC) $(CFLAGS) tests/test_vecadd.c -L$(LIB) -lbitnet_vecadd -o $(BUILD)/test_vecadd -lm
	$(CC) $(CFLAGS) tests/test_rmsnorm.c -L$(LIB) -lbitnet_rmsnorm -o $(BUILD)/test_rmsnorm -lm
	$(CC) $(CFLAGS) tests/test_softmax.c -L$(LIB) -lbitnet_softmax -o $(BUILD)/test_softmax -lm
	$(CC) $(CFLAGS) tests/test_rope.c -L$(LIB) -lbitnet_rope -o $(BUILD)/test_rope -lm
	$(CC) $(CFLAGS) tests/test_attention.c -L$(LIB) -lbitnet_attention -o $(BUILD)/test_attention -lm
	$(CC) $(CFLAGS) tests/test_activate.c -L$(LIB) -lbitnet_activate -o $(BUILD)/test_activate -lm
	$(CC) $(CFLAGS) tests/test_fused_attn.c -L$(LIB) -lbitnet_fused_attn -o $(BUILD)/test_fused_attn -lm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_i2s
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_quant
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_lut
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_vecadd
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_rmsnorm
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_softmax
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_rope
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_attention
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_activate
	LD_LIBRARY_PATH=$(LIB) $(BUILD)/test_fused_attn

clean:
	rm -rf $(BUILD)
