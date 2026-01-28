
#ifndef IRIS_SDK_EXPORT_H
#define IRIS_SDK_EXPORT_H

#ifdef IRIS_SDK_STATIC_DEFINE
#  define IRIS_SDK_EXPORT
#  define IRIS_SDK_NO_EXPORT
#else
#  ifndef IRIS_SDK_EXPORT
#    ifdef iris_sdk_EXPORTS
        /* We are building this library */
#      define IRIS_SDK_EXPORT 
#    else
        /* We are using this library */
#      define IRIS_SDK_EXPORT 
#    endif
#  endif

#  ifndef IRIS_SDK_NO_EXPORT
#    define IRIS_SDK_NO_EXPORT 
#  endif
#endif

#ifndef IRIS_SDK_DEPRECATED
#  define IRIS_SDK_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef IRIS_SDK_DEPRECATED_EXPORT
#  define IRIS_SDK_DEPRECATED_EXPORT IRIS_SDK_EXPORT IRIS_SDK_DEPRECATED
#endif

#ifndef IRIS_SDK_DEPRECATED_NO_EXPORT
#  define IRIS_SDK_DEPRECATED_NO_EXPORT IRIS_SDK_NO_EXPORT IRIS_SDK_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef IRIS_SDK_NO_DEPRECATED
#    define IRIS_SDK_NO_DEPRECATED
#  endif
#endif

#endif /* IRIS_SDK_EXPORT_H */
