# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(embree-download NONE)

include(ExternalProject)
ExternalProject_Add(embree-download
                    GIT_CONFIG advice.detachedHead=false;URL;https://github.com/embree/embree/archive/v2.17.4.tar.gz;URL_MD5;2038f3216b1d626e87453aee72c470e5
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/embree"
                    BINARY_DIR          "D:/Coding/libigl/build/embree-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
