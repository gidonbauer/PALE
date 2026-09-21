TARGETS = ${addprefix bin/, ${basename ${notdir ${wildcard examples/*.cpp}}}}
POISFFT_TARGETS = bin/advection-diffusion bin/mac bin/lid-driven-cavity bin/taylor-green
GSL_TARGETS = bin/scriven
NON_POISFFT_TARGETS = ${filter-out ${POISFFT_TARGETS} ${GSL_TARGETS}, ${TARGETS}}
HEADERS = ${wildcard src/*.hpp}

include Makefiles/compiler_flags.mk
include Makefiles/libs.mk

all: ${TARGETS}

${NON_POISFFT_TARGETS}: bin/%: examples/%.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -o $@ $< ${CXX_LIB}

${POISFFT_TARGETS}: bin/%: examples/%.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${POISFFT_INC} -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

${GSL_TARGETS}: bin/%: examples/%.cpp ${HEADERS} | bin output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${GSL_INC} -o $@ $< ${CXX_LIB} ${GSL_LIB}

bin output: %:
	mkdir -p $@

clean:
	rm -fr bin

include test/test.mk
include bench/bench.mk

.PHONY: all clean
