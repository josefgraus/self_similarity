# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(triangle-download NONE)

include(ExternalProject)
ExternalProject_Add(triangle-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/triangle.git;GIT_TAG;d6761dd691e2e1318c83bf7773fea88d9437464a
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/triangle"
                    BINARY_DIR          "D:/Coding/libigl/build/triangle-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
