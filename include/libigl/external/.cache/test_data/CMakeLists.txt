# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(test_data-download NONE)

include(ExternalProject)
ExternalProject_Add(test_data-download
                    GIT_REPOSITORY;https://github.com/libigl/libigl-tests-data;GIT_TAG;c81bb3b3db4cfd78bac6d359d845c45bc1059c9a;GIT_CONFIG advice.detachedHead=false
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/../tests/data"
                    BINARY_DIR          "D:/Coding/libigl/build/test_data-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
