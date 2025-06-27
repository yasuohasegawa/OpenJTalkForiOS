#!/bin/bash

set -e

IOS_MIN_SDK_VERSION="13.0"
SCRIPT_DIR=$(pwd)
OUTPUT_DIR="${SCRIPT_DIR}/ios_build"

echo "--- Cleaning old build artifacts ---"
rm -rf "${OUTPUT_DIR}"
rm -rf hts_engine_api-1.10/build_*
rm -rf open_jtalk-1.11/build_*

mkdir -p "${OUTPUT_DIR}/lib"
mkdir -p "${OUTPUT_DIR}/include"

IPHONEOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
IPHONESIMULATOR_SDK_PATH=$(xcrun --sdk iphonesimulator --show-sdk-path)
if [ -z "$IPHONEOS_SDK_PATH" ] || [ -z "$IPHONESIMULATOR_SDK_PATH" ]; then
    echo "Error: Could not find iOS SDK paths." >&2
    exit 1
fi

build_hts_engine_manually() {
    echo "========================================"
    echo "Building HTS Engine MANUALLY"
    echo "========================================"
    
    local SRC_DIR="${SCRIPT_DIR}/hts_engine_api-1.10"; cd "${SRC_DIR}"; rm -rf build_*
    local HTS_SOURCES="lib/HTS_audio.c lib/HTS_engine.c lib/HTS_gstream.c lib/HTS_label.c lib/HTS_misc.c lib/HTS_model.c lib/HTS_pstream.c lib/HTS_sstream.c lib/HTS_vocoder.c"

    for ARCH in "arm64" "x86_64"; do
        if [ "$ARCH" == "arm64" ]; then SDK_PATH=$IPHONEOS_SDK_PATH; TARGET="arm64-apple-ios${IOS_MIN_SDK_VERSION}"; else SDK_PATH=$IPHONESIMULATOR_SDK_PATH; TARGET="x86_64-apple-ios${IOS_MIN_SDK_VERSION}-simulator"; fi

        echo "--- Compiling HTS Engine for ${ARCH} ---"
        local BUILD_DIR="${SRC_DIR}/build_${ARCH}"; mkdir -p "${BUILD_DIR}/obj" "${BUILD_DIR}/lib"
        local CFLAGS="-arch ${ARCH} -isysroot ${SDK_PATH} -target ${TARGET} -O3 -Wall -I./include"
        local COMPILER=$(xcrun -f clang); local ARCHIVER=$(xcrun -f ar)

        for SRC_FILE in $HTS_SOURCES; do OBJ_FILE="${BUILD_DIR}/obj/$(basename ${SRC_FILE} .c).o"; ${COMPILER} ${CFLAGS} -c ${SRC_FILE} -o ${OBJ_FILE}; done
        local LIB_FILE="${BUILD_DIR}/lib/libHTSEngine.a"; ${ARCHIVER} -rcs "${LIB_FILE}" "${BUILD_DIR}/obj"/*.o
    done

    echo "--- Creating universal HTS Engine library ---"
    lipo -create "${SRC_DIR}/build_arm64/lib/libHTSEngine.a" "${SRC_DIR}/build_x86_64/lib/libHTSEngine.a" -output "${OUTPUT_DIR}/lib/libHTSEngine.a"
    cp -R "${SRC_DIR}/include/." "${OUTPUT_DIR}/include/"
    cd "${SCRIPT_DIR}"
}

build_openjtalk_manually() {
    echo "========================================"
    echo "Building Open JTalk MANUALLY"
    echo "========================================"
    
    local SRC_DIR="${SCRIPT_DIR}/open_jtalk-1.11"; cd "${SRC_DIR}"; rm -rf build_*

    echo "--- Creating custom config.h for iOS build ---"
    cat <<EOF > mecab/src/config.h
#ifndef MECAB_CONFIG_H
#define MECAB_CONFIG_H
#define HAVE_CONFIG_H 1
#define HAVE_STDINT_H 1
#define MECAB_DEFAULT_RC "/dev/null"
#define DIC_VERSION 102
#define CHARSET_UTF_8 1
#define PACKAGE "open-jtalk"
#define VERSION "1.11"
#endif // MECAB_CONFIG_H
EOF

    echo "--- Patching legacy source files ---"
    sed -i.bak 's/#include <string>/#include <string>\n#include <fcntl.h>\n#include <unistd.h>\n#include <sys\/stat.h>/' mecab/src/mmap.h
    sed -i.bak 's/flag | O_BINARY/flag/g' mecab/src/mmap.h
    sed -i.bak 's/#include "common.h"/#include "common.h"\n#include <dirent.h>/' mecab/src/utils.h

    local C_SOURCES="jpcommon/jpcommon.c jpcommon/jpcommon_label.c jpcommon/jpcommon_node.c mecab2njd/mecab2njd.c njd/njd.c njd/njd_node.c njd2jpcommon/njd2jpcommon.c njd_set_accent_phrase/njd_set_accent_phrase.c njd_set_accent_type/njd_set_accent_type.c njd_set_digit/njd_set_digit.c njd_set_long_vowel/njd_set_long_vowel.c njd_set_pronunciation/njd_set_pronunciation.c njd_set_unvoiced_vowel/njd_set_unvoiced_vowel.c text2mecab/text2mecab.c"
    local CPP_SOURCES="mecab/src/mecab.cpp mecab/src/dictionary_rewriter.cpp mecab/src/char_property.cpp mecab/src/connector.cpp mecab/src/context_id.cpp mecab/src/dictionary.cpp mecab/src/eval.cpp mecab/src/feature_index.cpp mecab/src/iconv_utils.cpp mecab/src/lbfgs.cpp mecab/src/learner.cpp mecab/src/libmecab.cpp mecab/src/nbest_generator.cpp mecab/src/param.cpp mecab/src/string_buffer.cpp mecab/src/tagger.cpp mecab/src/tokenizer.cpp mecab/src/utils.cpp mecab/src/viterbi.cpp mecab/src/writer.cpp"

    for ARCH in "arm64" "x86_64"; do
        if [ "$ARCH" == "arm64" ]; then SDK_PATH=$IPHONEOS_SDK_PATH; TARGET="arm64-apple-ios${IOS_MIN_SDK_VERSION}"; else SDK_PATH=$IPHONESIMULATOR_SDK_PATH; TARGET="x86_p64-apple-ios${IOS_MIN_SDK_VERSION}-simulator"; fi

        echo "--- Compiling Open JTalk for ${ARCH} ---"
        local BUILD_DIR="${SRC_DIR}/build_${ARCH}"; mkdir -p "${BUILD_DIR}/obj" "${BUILD_DIR}/lib"
        local COMPILER_C=$(xcrun -f clang); local COMPILER_CXX=$(xcrun -f clang++); local ARCHIVER=$(xcrun -f ar)
        
        local FORCE_INCLUDE_FLAG="-include mecab/src/config.h"
        local BASE_FLAGS="-arch ${ARCH} -isysroot ${SDK_PATH} -target ${TARGET} -O3 -Wall -Wno-deprecated ${FORCE_INCLUDE_FLAG} -I${OUTPUT_DIR}/include -I./mecab/src -I./jpcommon -I./njd -I./text2mecab -I./mecab2njd -I./njd_set_pronunciation -I./njd_set_digit -I./njd_set_accent_phrase -I./njd_set_accent_type -I./njd_set_unvoiced_vowel -I./njd_set_long_vowel -I./njd2jpcommon"
        local CFLAGS="${BASE_FLAGS} -DCHAR_INF=65535 -DCOST_MAX=65535"; local CXXFLAGS="${BASE_FLAGS} -DCHAR_INF=65535 -DCOST_MAX=65535"
        
        echo "   Compiling C sources..."; for SRC_FILE in $C_SOURCES; do OBJ_FILE="${BUILD_DIR}/obj/$(basename ${SRC_FILE} .c).o"; ${COMPILER_C} ${CFLAGS} -c ${SRC_FILE} -o ${OBJ_FILE}; done
        echo "   Compiling C++ sources..."; for SRC_FILE in $CPP_SOURCES; do OBJ_FILE="${BUILD_DIR}/obj/$(basename ${SRC_FILE} .cpp).o"; ${COMPILER_CXX} ${CXXFLAGS} -c ${SRC_FILE} -o ${OBJ_FILE}; done

        local LIB_FILE="${BUILD_DIR}/lib/libopenjtalk.a"; echo "   Archiving all object files..."; ${ARCHIVER} -rcs "${LIB_FILE}" "${BUILD_DIR}/obj"/*.o
    done

    echo "--- Creating universal Open JTalk library ---"
    lipo -create "${SRC_DIR}/build_arm64/lib/libopenjtalk.a" "${SRC_DIR}/build_x86_64/lib/libopenjtalk.a" -output "${OUTPUT_DIR}/lib/libopenjtalk.a"

    echo "--- Copying and creating headers ---"
    mkdir -p "${OUTPUT_DIR}/include/open_jtalk"
    cp mecab/src/*.h jpcommon/*.h njd/*.h text2mecab/*.h mecab2njd/*.h njd2jpcommon/*.h njd_set_pronunciation/*.h njd_set_digit/*.h njd_set_accent_phrase/*.h njd_set_accent_type/*.h njd_set_unvoiced_vowel/*.h njd_set_long_vowel/*.h "${OUTPUT_DIR}/include/open_jtalk/"
    cp mecab/src/config.h "${OUTPUT_DIR}/include/open_jtalk/"; cd "${SCRIPT_DIR}"
}

build_hts_engine_manually
build_openjtalk_manually

echo "-------------------------------------------"; echo "BUILD COMPLETE!"; echo "Your universal libraries and headers are in:"; echo "${OUTPUT_DIR}"; echo "-------------------------------------------"