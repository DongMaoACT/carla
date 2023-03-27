// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <cstdint>

namespace carla {
namespace image {
namespace detail {

    static constexpr
#if __cplusplus >= 201703L // C++17
    inline
#endif
    // Please update documentation if you change this.
    uint8_t CITYSCAPES_PALETTE_MAP[][3u] = {
        {96u,   96u,  96u},      // unlabeled     =   0u
        {192u,  192u, 192u},     // road          =   1u
        {96u,   96u,  96u},      // sidewalk      =   2u
        {96u,   96u,  96u},      // building      =   3u
        {96u,   96u,  96u},      // wall          =   4u
        {96u,   96u,  96u},      // fence         =   5u
        {96u,   96u,  96u},      // pole          =   6u
        {96u,   96u,  96u},      // traffic light =   7u
        {96u,   96u,  96u},      // traffic sign  =   8u
        {96u,   96u,  96u},      // vegetation    =   9u
        {96u,   96u,  96u},      // terrain       =  10u
        {96u,   96u,  96u},      // sky           =  11u
        {0u,    0u,   0u},       // pedestrian    =  12u
        {0u,    0u,   0u},       // rider         =  13u
        {0u,    0u,   0u},       // Car           =  14u
        {0u,    0u,   0u},       // truck         =  15u
        {0u,    0u,   0u},       // bus           =  16u
        {0u,    0u,   0u},       // train         =  17u
        {0u,    0u,   0u},       // motorcycle    =  18u
        {0u,    0u,   0u},       // bicycle       =  19u
        {96u,   96u,  96u},      // static        =  20u
        {96u,   96u,  96u},      // dynamic       =  21u
        {96u,   96u,  96u},      // other         =  22u
        {96u,   96u,  96u},      // water         =  23u
        {96u,   96u,  96u},      // road line     =  24u
        {96u,   96u,  96u},      // ground        =  25u
        {96u,   96u,  96u},      // bridge        =  26u
        {96u,   96u,  96u},      // rail track    =  27u
        {96u,   96u,  96u},      // guard rail    =  28u
      };

} // namespace detail

  class CityScapesPalette {
  public:

    static constexpr auto GetNumberOfTags() {
      return sizeof(detail::CITYSCAPES_PALETTE_MAP) /
          sizeof(*detail::CITYSCAPES_PALETTE_MAP);
    }

    /// Return an RGB uint8_t array.
    ///
    /// @warning It overflows if @a tag is greater than GetNumberOfTags().
    static constexpr auto GetColor(uint8_t tag) {
      return detail::CITYSCAPES_PALETTE_MAP[tag % GetNumberOfTags()];
    }
  };

} // namespace image
} // namespace carla
