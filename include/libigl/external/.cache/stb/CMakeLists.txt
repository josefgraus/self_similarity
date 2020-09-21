# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(stb-download NONE)

include(ExternalProject)
ExternalProject_Add(stb-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/libigl-stb.git;GIT_TAG;e671ceb3def5e7029a23de14c55dc16301ad4dab
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/stb"
                    BINARY_DIR          "D:/Coding/libigl/build/stb-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
