macro(__dgl_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

#######################################################
# An option to specify the build type for a feature.
# Usage:
#   dgl_feature_option(<option_variable> "doc string" "dev" "release")
macro(dgl_feature_option variable description)
  set(__value "")
  foreach(arg ${ARGN})
    if(arg STREQUAL "dev" OR arg STREQUAL "test" OR arg STREQUAL "release")
      list(APPEND __value ${arg})
    endif()
  endforeach()

  if(${BUILD_TYPE} IN_LIST __value)
    __dgl_option(${variable} "${description}" ON)
  else()
    __dgl_option(${variable} "${description}" OFF)
  endif()
endmacro()

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   dgl_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(dgl_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __dgl_option(${variable} "${description}" ON)
      else()
        __dgl_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __dgl_option(${variable} "${description}" ON)
      else()
        __dgl_option(${variable} "${description}" OFF)
      endif()
    else()
      __dgl_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()
