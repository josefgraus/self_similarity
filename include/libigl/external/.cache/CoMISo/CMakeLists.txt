# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(CoMISo-download NONE)

include(ExternalProject)
ExternalProject_Add(CoMISo-download
                    GIT_CONFIG advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/CoMISo.git;GIT_TAG;fea3ee0ba7d42ee3eca202d484e4fad855e4d6aa
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/CoMISo"
                    BINARY_DIR          "D:/Coding/libigl/build/CoMISo-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
