TARGETS = ${addprefix bin/, ${basename ${notdir ${wildcard examples/*.cpp}}}}
POISFFT_TARGETS = bin/advection-diffusion bin/mac bin/lid-driven-cavity bin/taylor-green
GSL_TARGETS = bin/scriven bin/scriven-2d

HEADERS = ${wildcard src/*.hpp}

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

${POISFFT_TARGETS}: EXTRA_INC += ${POISFFT_INC}
${POISFFT_TARGETS}: EXTRA_LIB += ${POISFFT_LIB}

${GSL_TARGETS}: EXTRA_INC += ${GSL_INC}
${GSL_TARGETS}: EXTRA_LIB += ${GSL_LIB}

${TARGETS}: bin/%: examples/%.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${EXTRA_INC} -o $@ $< ${CXX_LIB} ${EXTRA_LIB}

bin output: %:
	mkdir -p $@

clean:
	rm -fr bin

include test/test.mk
include bench/bench.mk

.PHONY: all clean
