cmake_minimum_required(VERSION 3.10)

project(lite3_sdk_deploy)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 警告忽略和编译选项
add_definitions(-w)
add_compile_options(-fPIC)

# 启用 backward-cpp 用于段错误排查
# 定义 USE_SIMULATION 宏以启用 main.cpp 中的 backward 代码
# 定义 BACKWARD_HAS_DW=1 以使用 libdw 进行符号解析

# add_definitions(-DUSE_SIMULATION -DBACKWARD_HAS_DW=1)

# 添加调试信息以便 backward 显示行号

# add_compile_options(-g -rdynamic)

# 配置选项
set(BUILD_PLATFORM "x86" CACHE STRING "Target platform (x86 or arm)")

if(BUILD_PLATFORM STREQUAL "arm")
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++") 
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(drdds REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# Use the same native MuJoCo library/header version as the Python environment.
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)"
  OUTPUT_VARIABLE MUJOCO_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MUJOCO_IMPORT_RESULT)
if(NOT MUJOCO_IMPORT_RESULT EQUAL 0)
  message(FATAL_ERROR "Install mpc/requirements.txt for the selected Python interpreter")
endif()
file(GLOB MUJOCO_LIBRARIES "${MUJOCO_ROOT}/libmujoco.so.*")
list(LENGTH MUJOCO_LIBRARIES MUJOCO_LIBRARY_COUNT)
if(NOT MUJOCO_LIBRARY_COUNT EQUAL 1)
  message(FATAL_ERROR "Expected one native MuJoCo library in ${MUJOCO_ROOT}")
endif()
list(GET MUJOCO_LIBRARIES 0 MUJOCO_LIBRARY)
add_executable(mpc_srbd mpc/native/mpc_srbd_node.cpp)
target_include_directories(mpc_srbd PRIVATE ${MUJOCO_ROOT}/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
target_link_libraries(mpc_srbd pybind11::embed ${MUJOCO_LIBRARY})
ament_target_dependencies(mpc_srbd rclcpp ament_index_cpp std_msgs geometry_msgs)
set_target_properties(mpc_srbd PROPERTIES INSTALL_RPATH "${MUJOCO_ROOT}")
install(TARGETS mpc_srbd DESTINATION lib/${PROJECT_NAME})

set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
  include/types
  include/utils
  interface/robot
  interface/user_command
  state_machine
  run_policy
  ${THIRD_PARTY}/eigen/
  ${THIRD_PARTY}/gamepad/include
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/include
)

# 子目录
add_subdirectory(interface)

# 运行时路径设置
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


set(CMAKE_INSTALL_RPATH 
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/lib/
  ../lib/onnxruntime/
)

file(GLOB_RECURSE STATE_MACHINE_SRC "state_machine/*.c*")

add_executable(rl_deploy main.cpp ${STATE_MACHINE_SRC})

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${THIRD_PARTY}/onnxruntime//${BUILD_PLATFORM}/lib/libonnxruntime.so")

ament_target_dependencies(rl_deploy
  ament_index_cpp
  rclcpp
  drdds
  std_msgs
  geometry_msgs
)
if(BUILD_PLATFORM STREQUAL "arm")
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
    # -ldw  启用backward调试功能需要libdw库
  )
else()
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
    -levdev
    # -ldw  启用backward调试功能需要libdw库
  )
endif()

install(TARGETS rl_deploy
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mpc/upstream/mpx/jax_ocp_solvers/jax_ocp_solvers/optimizers.py")
  message(FATAL_ERROR "MPX dependency missing from src/Lite3_sdk_deploy/mpc/upstream")
endif()

install(DIRECTORY mpc/ DESTINATION lib/${PROJECT_NAME}/mpc USE_SOURCE_PERMISSIONS
  PATTERN "__pycache__" EXCLUDE PATTERN "tests" EXCLUDE
  PATTERN ".git" EXCLUDE)
install(DIRECTORY Lite3_description DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mpc/tests/native_benchmark.cpp"
    AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mpc/tests/native_mapper_probe.cpp")
  add_executable(native_benchmark mpc/tests/native_benchmark.cpp)
  target_include_directories(native_benchmark PRIVATE ${MUJOCO_ROOT}/include)
  target_link_libraries(native_benchmark pybind11::embed ${MUJOCO_LIBRARY})
  add_executable(native_mapper_probe mpc/tests/native_mapper_probe.cpp)
  target_include_directories(native_mapper_probe PRIVATE ${MUJOCO_ROOT}/include)
  target_link_libraries(native_mapper_probe ${MUJOCO_LIBRARY})
endif()

ament_package()
