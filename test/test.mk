DEFAULT_BUILD_TESTS = test/Advection-Cartesian.cpp \
                      test/Advection-Polar.cpp \
                      test/Iterator.cpp \
                      test/Polar-Couette.cpp \
                      test/Polar-Channel.cpp \
                      test/Boundary.cpp \
                      test/Multigrid-Spherical.cpp \
                      test/Hill-Vortex \
                      test/ALE-Polar-Conservation \
                      test/ALE-Spherical-Conservation \
                      test/GCL \
                      test/Scalar-Source \
                      test/IB-Polar-Channel
DEFAULT_BUILD_TESTS := ${addprefix bin/, ${basename ${DEFAULT_BUILD_TESTS}}}

${DEFAULT_BUILD_TESTS}: bin/test/%: test/%.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -o $@ $< ${CXX_LIB}

bin/test/Multigrid: test/Multigrid.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${POISFFT_INC} -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

bin/test/Taylor-Green-FFT: test/Taylor-Green.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${POISFFT_INC} -DFFT_POISSON=1 -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

bin/test/Taylor-Green-MG: test/Taylor-Green.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -DMG_POISSON=1 -o $@ $< ${CXX_LIB}

bin/test/Channel-FFT: test/Channel.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${POISFFT_INC} -DFFT_POISSON=1 -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

bin/test/Channel-MG: test/Channel.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -DMG_POISSON=1 -o $@ $< ${CXX_LIB}

bin/test/Scriven-2D: test/Scriven.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${GSL_INC} -DDIMENSION=2 -o $@ $< ${CXX_LIB} ${GSL_LIB}

bin/test/Scriven-3D: test/Scriven.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${GSL_INC} -DDIMENSION=3 -o $@ $< ${CXX_LIB} ${GSL_LIB}

bin/test/ALE-Scalar-Source: test/Scalar-Source.cpp  ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -DUSE_ALE -o $@ $< ${CXX_LIB}

bin/test/IB-Channel-FFT: test/IB-Channel.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} ${POISFFT_INC} -DFFT_POISSON=1 -o $@ $< ${CXX_LIB} ${POISFFT_LIB}

bin/test/IB-Channel-MG: test/IB-Channel.cpp ${HEADERS} | bin/test test/output
	${CXX} ${CXX_FLAGS} ${CXX_INC} -DMG_POISSON=1 -o $@ $< ${CXX_LIB}

test/output bin/test: %:
	mkdir -p $@

test-clean:
	rm -r test/logs test/output

.PHONY: test-clean
