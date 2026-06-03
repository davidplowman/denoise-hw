/*=============================================================================
Copyright (c) 2018 Raspberry Pi Trading Ltd.
=============================================================================*/

// NOTE: compile with: g++ -O2 -shared -fPIC -o cdn_bayer.so cdn_bayer.cpp

#include <iostream>
#include <assert.h>
#include <inttypes.h>
#include "cdn_bayer.hpp"
#include "image.hpp"

static const int G_ADJUST_PRECISION = 8;

static const uint16_t DEFAULT_THRESHOLD = 1000;
static const uint8_t DEFAULT_IIR_STRENGTH = 48;
static const uint8_t DEFAULT_G_ADJUST = 0;
static const int FIR_FILTER_TERMS = 21;

CdnBayer::CdnBayer() : thresh(DEFAULT_THRESHOLD), iir_strength(DEFAULT_IIR_STRENGTH), g_adjust(DEFAULT_G_ADJUST)
{
}

void CdnBayer::configure(pisp_be_cdn_config const &config)
{
  order = (BayerOrder)config.bayer_order;
  thresh = config.thresh;
  iir_strength = config.iir_strength;
  g_adjust = config.g_adjust;
}

void CdnBayer::getConfig(pisp_be_cdn_config &config) const
{
  config.thresh = thresh;
  config.iir_strength = iir_strength;
  config.g_adjust = g_adjust;
}

template <typename T> static uint16_t approx_log(T x)
{
    // Output 8 bits of mantissa plus 8 bits of exponent.
    int exp = -1;
    for (T tmp = x; tmp; tmp >>= 1, exp++);
    uint32_t mantissa = (exp > 8 ? x>>(exp-8) : x<<(8-exp))&0xff;
    return (exp<<8) | mantissa;
}

static uint32_t approx_exp(uint16_t x)
{
    int exp = x >> 8;
    uint32_t mantissa = x & 0xff;
    if (exp > 8) mantissa <<= (exp - 8);
    else if (exp < 8) mantissa >>= (8 - exp);
    return (1<<exp) | mantissa;
}

static uint32_t approx_div(uint64_t a, uint16_t log_b)
{
  return approx_exp(approx_log<uint64_t>(a)-log_b);
}

Image filter_image(Image const &src, int thresh, int iir_strength)
{
  Image dst(src.getWidth(), src.getHeight(), 2, 32);
  int h = dst.getHeight();
  int w = dst.getWidth();
  int t = std::max(thresh, 1); // u16

  Image buf(src.getWidth(), 1, 3, 32); // but in reality channel 0 is u16, and channels 1 and 2 are s17
  for (int i = 0; i < w; i++) {
    buf.pix<int>(0,i,0) = src.pix<int>(0,i,0);
    buf.pix<int>(0,i,1) = src.pix<int>(0,i,1);
    buf.pix<int>(0,i,2) = src.pix<int>(0,i,2);
  }
  const int SIZE_X = 12;
  const int STEP_X = 4;
  const int SIZE_Y = 4;
  const int STEP_Y = 2;
  static_assert(FIR_FILTER_TERMS == ((SIZE_X/STEP_X)*2+1)*(SIZE_Y/STEP_Y+1), "FIR_FILTER_TERMS incorrect");

  // This is a bilateral filter with a large (particularly, wide) but sparse kernel. The vertical context
  // is much smaller; we make up for this to some extent with an IIR filter term that gets included.
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int g = src.pix<int>(y,x,0); // u16
      int cr = src.pix<int>(y,x,1), cb = src.pix<int>(y,x,2); // both s17
      int64_t Sg_wd = 0, Scr_wd = 0, Scb_wd = 0;
      uint32_t Sw = 0;
      // First add the FIR terms, these are placed at and below the current pixel to reduce any phase shift from the IIR term.
      for (int j = 0; j <= SIZE_Y; j+=STEP_Y) {
        if (y+j >= 0 && y+j < h) {
          for (int i = -SIZE_X; i <= SIZE_X; i+=STEP_X) {
            if (x+i >= 0 && x+i < w) {
              int dg = src.pix<int>(y+j,x+i,0) - g;  // s17
              int dcr = src.pix<int>(y+j,x+i,1) - cr, dcb = src.pix<int>(y+j,x+i,2) - cb; // both s18
              int s = abs(dg)+abs(dcr)+abs(dcb);  // up to s20
              int wt = std::min(t, std::max(0, std::max(9*t/2-3*s, 2*t-s)/4)); // result here is u16
              Sg_wd += wt*dg, Scr_wd += wt*dcr, Scb_wd += wt*dcb, Sw += wt;
            }
          }
        }
      }
      // There are 21 terms added so far, so:
      // Sg_wd is 21 lots of u16 * s17 which is s38
      // Scr_wd and Scb_wd are 21 lots of u16 * s18 which is s39
      // Sw is 21 lots of u16, or u21.
      if (Sg_wd < -(((int64_t)1)<<37) || Sg_wd >= (((int64_t)1)<<37)) throw std::runtime_error("CdnBayer: Sg_wd overflow");
      if (Scr_wd < -(((int64_t)1)<<38) || Scr_wd >= (((int64_t)1)<<38)) throw std::runtime_error("CdnBayer: Scr_wd overflow");
      if (Scb_wd < -(((int64_t)1)<<38) || Scb_wd >= (((int64_t)1)<<38)) throw std::runtime_error("CdnBayer: Scb_wd overflow");
      // Now much the same again, but for the IIR values.
      int prev_g = buf.pix<int>(0,x,0), prev_cr = buf.pix<int>(0,x,1), prev_cb = buf.pix<int>(0,x,2);
      int dg = prev_g - g, dcr = prev_cr - cr, dcb = prev_cb - cb;
      int s = abs(dg) + abs(dcr) + abs(dcb);
      int wt = std::min(t, std::max(0, std::max(9*t/2-3*s, 2*t-s)/4));
      wt *= iir_strength;
      // If we limit iir_strength to 255 - 21 (there being 21 terms added already) then:
      // Sg_wd has up to 256 lots of u16 * s17 which is s41
      // Scr_wd and Scb_wd are up to 256 lots of u16 * s18 which is s42
      // wt is 256 lots of u16, or u24.
      Sg_wd += ((int64_t)wt)*dg, Scr_wd += ((int64_t)wt)*dcr, Scb_wd += ((int64_t)wt)*dcb, Sw += wt;
      if (Sg_wd < -(((int64_t)1)<<40) || Sg_wd >= (((int64_t)1)<<40)) throw std::runtime_error("CdnBayer: Sg_wd overflow");
      if (Scr_wd < -(((int64_t)1)<<41) || Scr_wd >= (((int64_t)1)<<41)) throw std::runtime_error("CdnBayer: Scr_wd overflow");
      if (Scb_wd < -(((int64_t)1)<<41) || Scb_wd >= (((int64_t)1)<<41)) throw std::runtime_error("CdnBayer: Scb_wd overflow");
#if 0
      int deltay = Sg_wd / Sw, deltau = Scr_wd / Sw, deltav = Scb_wd / Sw;
      (void)approx_div;
#else
      int deltay = 0, deltau = 0, deltav = 0;
      uint16_t log_Sw = approx_log<uint32_t>(Sw); // Sw is u24 so this will fit in a u13 (5 exponent + 8 mantissa)
      // In theory deltay can't be more than s17, but the approx_div means we need to allow for an s18 result.
      // Similarly deltau and deltav will need to allow for an s19 result.
      if (Sg_wd >= Sw) deltay = approx_div(Sg_wd, log_Sw);
      else if (Sg_wd <= -(int64_t)Sw) deltay = -approx_div(-Sg_wd, log_Sw);
      if (Scr_wd >= Sw) deltau = approx_div(Scr_wd, log_Sw);
      else if (Scr_wd <= -(int64_t)Sw) deltau = -approx_div(-Scr_wd, log_Sw);
      if (Scb_wd >= Sw) deltav = approx_div(Scb_wd, log_Sw);
      else if (Scb_wd <= -(int64_t)Sw) deltav = -approx_div(-Scb_wd, log_Sw);
      if (deltay < -(1<<17) || deltay >= (1<<17)) throw std::runtime_error("CdnBayer: deltay overflow");
      if (deltau < -(1<<18) || deltau >= (1<<18)) throw std::runtime_error("CdnBayer: deltau overflow");
      if (deltav < -(1<<18) || deltav >= (1<<18)) throw std::runtime_error("CdnBayer: deltav overflow");
#endif
      g += deltay, cr += deltau, cb += deltav;
      // Again, the approx_div means these need to be clipped back to their nominal ranges.
      g = std::max(0, std::min(g, 65535));
      cr = std::max(-65536, std::min(cr, 65535));
      cb = std::max(-65536, std::min(cb, 65535));
      // Update the line buffer with the new IIR values, and our output "image".
      buf.pix<int>(0,x,0) = g, buf.pix<int>(0,x,1) = cr, buf.pix<int>(0,x,2) = cb;
      dst.pix<int>(y,x,0) = cr, dst.pix<int>(y,x,1) = cb;
    }
  }
  return dst;
}

void CdnBayer::doFrame(Image & dst, Image const & src)
{
  assert(src.getBitsPerSample() == 16 && src.getChannels() == 1);
  dst.configure(src);
  int w = src.getWidth(), h = src.getHeight();
  assert(w%2 == 0 && h%2 == 0);
  int r_idx, gr_idx, gb_idx, b_idx;
  image_decode_bayer(order, r_idx, gr_idx, gb_idx, b_idx);

  // This image contains ints even though we only need 16bpp for channel 0 (average g) and
  // 17bpp for channels 1 and 2 (r minus g and b minus g).
  Image g_chroma(w/2, h/2, 5, 32);
  for (int y = 0; y < h; y+=2) {
      for (int x = 0; x < w; x+=2) {
          int gr_avg = ((int)src.pixClip<uint16_t>(Image::CLIP_BAYER,y+(r_idx>>1),x+(r_idx&1)-1) +
            src.pixClip<uint16_t>(Image::CLIP_BAYER,y+(r_idx>>1),x+(r_idx&1)+1)) >> 1;
          int gb_avg = ((int)src.pixClip<uint16_t>(Image::CLIP_BAYER,y+(b_idx>>1),x+(b_idx&1)-1) +
            src.pixClip<uint16_t>(Image::CLIP_BAYER,y+(b_idx>>1),x+(b_idx&1)+1)) >> 1;
          g_chroma.pix<int>(y/2,x/2,0) = (gr_avg + gb_avg) >> 1;
          g_chroma.pix<int>(y/2,x/2,1) = (int)src.pix<uint16_t>(y+(r_idx>>1),x+(r_idx&1)) - gr_avg;
          g_chroma.pix<int>(y/2,x/2,2) = (int)src.pix<uint16_t>(y+(b_idx>>1),x+(b_idx&1)) - gb_avg;
          g_chroma.pix<int>(y/2,x/2,3) = gr_avg; // filter_image doesn't use these, but save them for the reconstruction afterwards
          g_chroma.pix<int>(y/2,x/2,4) = gb_avg;
      }
  }

  // We get back a 2-channel image with the new r-g (channel 0) and b-g (channel 1) values.
  // Again, these two channels are signed 17-bits.
  Image new_chroma = filter_image(g_chroma, thresh, iir_strength);

  // Make our output image reflect the new r-g and b-g values. Optionally, according to g_adjust, don't
  // make all the change go into r and b, but allow some of it to go into g.
  for (int y = 0; y < h; y+=2) {
      for (int x = 0; x < w; x+=2) {
          int r = src.pix<uint16_t>(y+(r_idx>>1),x+(r_idx&1));
          int gr = src.pix<uint16_t>(y+(gr_idx>>1),x+(gr_idx&1));
          int gb = src.pix<uint16_t>(y+(gb_idx>>1),x+(gb_idx&1));
          int b = src.pix<uint16_t>(y+(b_idx>>1),x+(b_idx&1));
          int r_minus_g = new_chroma.pix<int>(y/2,x/2,0); // s17
          int b_minus_g = new_chroma.pix<int>(y/2,x/2,1); // s17
          int gr_avg = g_chroma.pix<int>(y/2,x/2,3);      // u16
          int gb_avg = g_chroma.pix<int>(y/2,x/2,4);      // u16
          int delta_r = r_minus_g + gr_avg - r;
          int delta_b = b_minus_g + gb_avg - b;
          int delta_g = ((delta_r + (delta_b>>1)) * g_adjust) >> G_ADJUST_PRECISION;
          gr -= delta_g, gb -= delta_g, gr_avg -= delta_g, gb_avg -= delta_g;
          r = r_minus_g + gr_avg, b = b_minus_g + gb_avg;
          dst.pix<uint16_t>(y+(r_idx>>1),x+(r_idx&1)) = std::max(0, std::min(65535, r));
          dst.pix<uint16_t>(y+(gr_idx>>1),x+(gr_idx&1)) = std::max(0, std::min(65535, gr));
          dst.pix<uint16_t>(y+(gb_idx>>1),x+(gb_idx&1)) = std::max(0, std::min(65535, gb));
          dst.pix<uint16_t>(y+(b_idx>>1),x+(b_idx&1)) = std::max(0, std::min(65535, b));
      }
  }
}

extern "C"
void run_cdn(uint16_t *buffer, int width, int height, int stride_bytes,
             int thresh, int iir_strength, int g_adjust, int bayer_order)
{
    Image src(width, height, 1, 16);
    Image dest(width, height, 1, 16);
    pisp_be_cdn_config config;
    CdnBayer stage;

    for (int j = 0; j < height; j++)
        memcpy(src.ptr<uint16_t>(j), (uint8_t *)buffer + j * stride_bytes, width * 2);

    config.thresh = thresh;
    config.iir_strength = iir_strength;
    config.g_adjust = g_adjust;
    config.bayer_order = bayer_order;

    stage.configure(config);
    stage.doFrame(dest, src);

    for (int j = 0; j < height;j++)
        memcpy((uint8_t *)buffer + j * stride_bytes, dest.ptr<uint16_t>(j), width * 2);
}