# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(tetgen-download NONE)

include(ExternalProject)
ExternalProject_Add(tetgen-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/tetgen.git;GIT_TAG;d2dcc33cb8551f16e302c8130ce12fa52992cd09
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/tetgen"
                    BINARY_DIR          "D:/Coding/libigl/build/tetgen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
