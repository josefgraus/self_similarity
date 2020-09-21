# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(googletest-download NONE)

include(ExternalProject)
ExternalProject_Add(googletest-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/google/googletest;GIT_TAG;release-1.8.1
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/googletest"
                    BINARY_DIR          "D:/Coding/libigl/build/googletest-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
