#pragma once

// windows dll export/import
#ifdef _WIN32
  #ifdef COLMAP_API_EXPORTS
    #define COLMAP_API __declspec(dllexport)
  #else
    #define COLMAP_API __declspec(dllimport)
  #endif
#else
  #define COLMAP_API
#endif
