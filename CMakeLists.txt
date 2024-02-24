# 各行の「#」より後はコメントになります．

# 使用するCMakeバージョンを指定します
cmake_minimum_required(VERSION 3.19)



# プロジェクトを定義します．
# プロジェクトはVisual Studioではソリューションに相当します．
# プロジェクトの中には複数の実行ファイルやライブラリを定義できます．
project(digitalcurling3_cnn  # プロジェクト名．Visual Studioではソリューション名になります．
    VERSION 0.1.0
    LANGUAGES CXX
)



# C++17標準を使用するためのコマンドです
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# DigitalCurling3ライブラリを読み込みます
add_subdirectory(extern/DigitalCurling3)

# TCP/IP接続に使用するBoost.Asioライブラリを読み込みます
# set(BOOST_DIR C:/path/to/boost)
# list(APPEND CMAKE_PREFIX_PATH ${BOOST_DIR})

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.74 REQUIRED COMPONENTS date_time regex)

# Threadsライブラリを読み込みます
find_package(Threads REQUIRED)

# libtorch
set(LIBTORCH_DIR /usr/local/libtorch/)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

# OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(include)

# 実行ファイルを定義します
add_executable(digitalcurling3_cnn  # 実行ファイルの名前はここの名前になります．なおプロジェクト名と一致させる必要はありません．
    main.cpp
    # ソースファイルやヘッダーファイルを追加する場合，ファイルを作成した後にファイル名をここに列挙します．
    # example.hpp
    # example.cpp
)

# リンクするライブラリを指定します
# インクルードディレクトリとリンクするライブラリファイルが設定されます
target_link_libraries(digitalcurling3_cnn  # add_executable() で実行ファイル名を変更した場合，同様にここも変更する必要があります．
  PRIVATE
    digitalcurling3::digitalcurling3
    Boost::headers
    Boost::date_time
    Boost::regex
    Threads::Threads
    "${TORCH_LIBRARIES}"
)

# Windowsで.dllファイルをコピー
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET digitalcurling3_cnn
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:digitalcurling3_cnn>)
endif (MSVC)
