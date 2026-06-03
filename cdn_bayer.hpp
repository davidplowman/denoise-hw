/*=============================================================================
Copyright (c) 2018 Raspberry Pi Trading Ltd.
=============================================================================*/

#ifndef _CDN_BAYER_HPP
#define _CDN_BAYER_HPP

#include <stdint.h>
#include "stage.hpp"
#include "image.hpp"

/**
 * struct pisp_be_cdn_config - PiSP Back End CDN config
 *
 * Colour Denoise configuration
 *
 * @thresh:		Constant for noise estimation
 * @iir_strength:	Relative strength of the IIR part of the filter
 * @g_adjust:		Proportion of the change assigned to the G channel
 */
struct pisp_be_cdn_config {
	uint16_t thresh;
	uint8_t iir_strength;
	uint8_t g_adjust;
    uint8_t bayer_order;
};

class CdnBayer : public Stage {
public:
  CdnBayer();
  ~CdnBayer() { }
  void configure(pisp_be_cdn_config const &config);
  void getConfig(pisp_be_cdn_config &config) const;
  void doFrame(Image & dst, Image const & src);
//private:
  BayerOrder order;
  uint16_t thresh;
  uint8_t iir_strength;
  uint8_t g_adjust;
};

#endif
