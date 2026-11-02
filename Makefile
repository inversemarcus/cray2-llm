# Build the CRAY-2 emulator (git submodule external/cray2sim) and run its self-tests.
all:
	$(MAKE) -C external/cray2sim

test:
	$(MAKE) -C external/cray2sim test

clean:
	$(MAKE) -C external/cray2sim clean

.PHONY: all test clean
