# The serving image keeps LLVM executables but removes development archives.
# ROCr's assembly-only build steps need just these imported tool targets.
if(NOT TARGET clang)
  add_executable(clang IMPORTED GLOBAL)
  set_target_properties(clang PROPERTIES IMPORTED_LOCATION "/opt/rocm/llvm/bin/clang")
endif()
