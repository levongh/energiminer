# Set necessary compile and link flags

# C++11 check and activation
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra -Wno-error=parentheses -pedantic -Wno-unused-parameter")

elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra -Wno-unused-parameter")

	if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -fcolor-diagnostics -Qunused-arguments")
	endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

	# declare Windows XP requirement
	# undefine windows.h MAX & MIN macros because they conflict with std::min & std::max functions
	# disable unsafe CRT Library functions warnings
	add_definitions(/D_WIN32_WINNT=0x0501 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS)

	# enable parallel compilation
	# specify Exception Handling Model
	# enable LTCG for faster builds
	# disable unknown pragma warnings (C4068)
	# disable conversion from 'size_t' to 'type', possible loss of data (C4267)
	# disable C++ exception specification ignored except to indicate a function is not __declspec(nothrow) (C4290)
	# disable decorated name length exceeded, name was truncated (C4503)
    # disable conversion from 'type1' to 'type2', possible loss of data
	add_compile_options(/MP /EHsc /GL /wd4068 /wd4267 /wd4290 /wd4503 /wd4244)

	# enable LTCG for faster builds
	set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")

	# enable LTCG for faster builds
	# enable unused references removal
	# enable RELEASE so that the executable file has its checksum set
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG /OPT:REF /OPT:ICF /RELEASE")
else ()
	message(WARNING "Your compiler is not tested, if you run into any issues, we'd welcome any patches.")
endif ()

set(SANITIZE NO CACHE STRING "Instrument build with provided sanitizer")
if(SANITIZE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${SANITIZE}")
endif()

