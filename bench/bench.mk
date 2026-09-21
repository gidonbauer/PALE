BENCH = ${addprefix bin/, ${basename ${wildcard bench/*.cpp}}}
POISFFT_BENCH = bin/bench/multigrid
NON_POISFFT_BENCH = ${filter-out ${POISFFT_BENCH}, ${BENCH}}

bench: ${BENCH}

BENCH_VTK_OUTPUT ?= 0
ifeq (${BENCH_VTK_OUTPUT}, 1)
	BENCH_FLAGS := -DPALE_BENCH_VTK_OUTPUT
else
	BENCH_FLAGS :=
endif

${NON_POISFFT_BENCH}: bin/bench/%: bench/%.cpp ${HEADERS} | bin/bench
	${CXX} ${CXX_FLAGS} ${BENCH_FLAGS} ${CXX_INC} -o $@ $< ${CXX_LIB}

bin/bench/multigrid: bench/multigrid.cpp ${HEADERS} | bin/bench
	${CXX} ${CXX_FLAGS} ${BENCH_FLAGS} ${CXX_INC} ${POISFFT_INC} -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

bin/bench:
	mkdir -p $@
