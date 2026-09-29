TARGETS = bin/advection-diffusion \
          bin/taylor-green \
          bin/lid-driven-cavity \
          bin/mac \
          bin/polar \
          bin/ale-polar \
          bin/ale-rotation \
          bin/old-scriven-3d \
          bin/scriven-2d \
          bin/scriven-3d

POISFFT_TARGETS = bin/advection-diffusion bin/mac bin/lid-driven-cavity bin/taylor-green
GSL_TARGETS = bin/old-scriven-3d bin/scriven-2d bin/scriven-3d

CUSTOM_TARGETS = bin/scriven-2d bin/scriven-3d
DEFAULT_TARGETS = ${filter-out ${CUSTOM_TARGETS}, ${TARGETS}}

HEADERS = ${wildcard src/*.hpp}

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

${POISFFT_TARGETS}: EXTRA_INC += ${POISFFT_INC}
${POISFFT_TARGETS}: EXTRA_LIB += ${POISFFT_LIB}

${GSL_TARGETS}: EXTRA_INC += ${GSL_INC}
${GSL_TARGETS}: EXTRA_LIB += ${GSL_LIB}

${DEFAULT_TARGETS}: bin/%: examples/%.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${EXTRA_INC} -o $@ $< ${CXX_LIB} ${EXTRA_LIB}

bin/scriven-2d: examples/scriven.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${EXTRA_INC} -DDIMENSION=2 -o $@ $< ${CXX_LIB} ${EXTRA_LIB}

bin/scriven-3d: examples/scriven.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${EXTRA_INC} -DDIMENSION=3 -o $@ $< ${CXX_LIB} ${EXTRA_LIB}

bin output: %:
	mkdir -p $@

clean:
	rm -fr bin

include test/test.mk
include bench/bench.mk

.PHONY: all clean
