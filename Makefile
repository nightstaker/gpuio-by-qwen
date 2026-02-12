# Simple Makefile for gpuio

CC = cc
CFLAGS = -Wall -Wextra -O2 -I include -pthread
LDFLAGS = -pthread

SRCS = src/gpuio.c
OBJS = $(SRCS:.c=.o)
LIB = libgpuio.a
TEST_SRCS = tests/unit/test_core.c
TEST_BIN = test_gpuio

.PHONY: all clean test

all: $(LIB) $(TEST_BIN)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(TEST_BIN): $(TEST_SRCS) $(LIB)
	$(CC) $(CFLAGS) -o $@ $< -L. -lgpuio $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(OBJS) $(LIB) $(TEST_BIN)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@
