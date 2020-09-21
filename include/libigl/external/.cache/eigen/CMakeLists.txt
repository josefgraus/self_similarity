# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 2.8.2)

project(eigen-download NONE)

include(ExternalProject)
ExternalProject_Add(eigen-download
                    GIT_CONFIG advice.detachedHead=false;URL;http://bitbucket.org/eigen/eigen/get/3.2.10.tar.gz;URL_MD5;8ad10ac703a78143a4062c9bda9d8fd3
                    SOURCE_DIR          "D:/Coding/libigl/cmake/../external/eigen"
                    BINARY_DIR          "D:/Coding/libigl/build/eigen-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    TLS_VERIFY          OFF
)
