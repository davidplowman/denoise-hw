/*=============================================================================
Copyright (c) 2017-2018 Raspberry Pi Trading Ltd.
=============================================================================*/

#ifndef _IMAGE_HPP
#define _IMAGE_HPP

#include <vector>
#include <algorithm>
#include <cstring>

  // Metadata. This is a grey area; it could be considered part of the
  // "parameters", but tends to stick to an image (though not included
  // in our Image class). Here it is restricted to Camera ID, exposure,
  // analogue gain, bit-depth and Bayer pattern.

  enum BayerOrder { // this matches VC_IMAGE_BAYER_ORDER_T
    RGGB = 0,
    GBRG,
    BGGR,
    GRBG
  };

  struct Metadata {
    char camName[128];  // Camera ID (from BRCM RAW) or "<make>/<model>" (from DNG)
    int  exposure;      // exposure time in microseconds; 0 if unknown
    int  againQ8;       // analogue gain used by sensor, scaled by 256
    int  sigbits;       // number of significant (MS) bits in raw data
    BayerOrder pattern; // CFA ordering of the raw image (we assume it is Bayer)
  };


// Our Image class is somewhat inspired by HDP's Image class definition
// and by OpenCV's Mat. ProtoISP needs different image formats from HDP:
// it requires single-channel images (Bayer), which HDP appears to lack,
// but no need for Alpha components. Since many images in protoISP are
// 16bpp, we bake-in support for different sample types (like OpenCV).
//
// NB: For now, all images are stored without padding and are interleaved.
// A more precise model will require padding and alignment, planar formats
// and perhaps MIPI-style packed formats for non-multiples of 8 bits(?)

class Image
{
  public:

    // Constructors and destructor
    Image() :
      width(0),
      height(0),
      channels(0),
      bps(0),
      data()
    {
    }

    Image(int _w, int _h, int _ch=1, int _bps=8) :
      width(_w),
      height(_h),
      channels(_ch),
      bps(_bps),
      data()
    {
      data.resize(_w*_h*_ch*(_bps>>3));
      std::fill(data.begin(), data.end(), 0);
    }

    ~Image() { }

    // Default constructors and assignment.
    Image (Image const &)             = default;
    Image (Image &&)                  = default;
    Image & operator= (Image const &) = default;
    Image & operator= (Image &&)      = default;

    // Re-use of existing image object; this will clear the content
    void configure(int _w, int _h, int _ch=1, int _bps=8)
    {
      width = _w;
      height = _h;
      channels = _ch;
      bps = _bps;
      data.resize(_w*_h*_ch*(_bps>>3));
      std::fill(data.begin(), data.end(), 0);
   }
    void configure(Image const & prototype) {
      if (&prototype != this)
        configure(prototype.width, prototype.height, prototype.channels, prototype.bps);
    }

    // Load a file in DNG, PNG or JPEG+BRCM formt; normalize to 16 bits per component
    bool load(char const * fileName, Metadata * pMeta = 0);

    // Save an image as either PNG or PGM/PPM depending on fileName.
    // 16-bit images will be saved with all 16 significant bits.
    bool save(char const * fileName) const;

    // Get dimensions
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    int getBitsPerSample() const { return bps; }
    int getBytesPerPixel() const { return (bps>>3)*channels; }
    int getStrideInBytes() const { return getBytesPerPixel()*width; }
    bool matches(Image const & rhs) const {
      return (width == rhs.width && height == rhs.height &&
              channels == rhs.channels && bps == rhs.bps);
    }

    // Pixel access
    template<class T> T * ptr(int row=0, int col=0, int ch=0) {
      return reinterpret_cast<T*>(&data[0]) + (row*width+col)*channels+ch;
    }
    template<class T> T const * ptr(int row=0, int col=0, int ch=0) const {
      return reinterpret_cast<T const *>(&data[0]) + (row*width+col)*channels+ch;
    }
    template<class T> T & pix(int row, int col, int ch=0) {
      return *ptr<T>(row,col,ch);
    }
    template<class T> T const & pix(int row, int col, int ch=0) const {
      return *ptr<T>(row,col,ch);
    }

    // A function that might come in useful, to access out-of-bounds pixels
    enum ClipType {
      CLIP_ZERO,
      CLIP_EDGE,
      CLIP_BAYER
    };
    template<class T> T pixClip(ClipType ct, int row, int col, int ch=0) const {
      if (row < 0 || row >= height || col < 0 || col >= width) {
        switch (ct) {
        case CLIP_EDGE:
          if (row < 0) row = 0;
          if (row >= height) row = height-1;
          if (col < 0) col = 0;
          if (col >= width) col = width-1;
          break;
        case CLIP_BAYER:
          if (row < 0) row &= 1;
          if (row >= height) row = height-2+((row^height)&1);
          if (col < 0) col &= 1;
          if (col >= width) col = width-2+((col^width)&1);
          break;
        default:
          return static_cast<T>(0);
        }
      }
      return pix<T>(row, col, ch);
    }

    void cropFrom(Image const & src, int x, int y, int w, int h)
    {
      if (x + w > src.width) w = src.width - x;
      if (y + h > src.height) h = src.height - y;
      if (x >= 0 && y >= 0 && w > 0 && h > 0) {
        int old_pitch = src.width*src.channels*(src.bps>>3);
        int new_pitch = w*src.channels*(src.bps>>3);
        if (this != &src) {
          channels = src.channels;
          bps = src.bps;
          data.resize(new_pitch * h);
        }
        unsigned char * dp = &(data[0]);
        unsigned char const * sp = (&(src.data[0])) + y*old_pitch + x*src.channels*(src.bps>>3);
        for(int i = 0; i < h; ++i) {
          if (dp != sp) std::memcpy(dp, sp, new_pitch);
          sp += old_pitch;
          dp += new_pitch;
        }
        width = w;
        height = h;
        data.resize(new_pitch * h);
      }
    }

  private:
    int width;
    int height;
    int channels;
    int bps;
    std::vector<unsigned char> data;
};

// Decode the Bayer type enum to give correct channel indices into an "image_debayer"ed image.

static inline void image_decode_bayer(BayerOrder order, int &r, int &gr, int &gb, int &b)
{
    if (order == RGGB)
        r = 0, gr = 1, gb = 2, b = 3;
    else if (order == GRBG)
        r = 1, gr = 0, gb = 3, b = 2;
    else if (order == GBRG)
        r = 2, gr = 3, gb = 0, b = 1;
    else
        r = 3, gr = 2, gb = 1, b = 0;
}

#endif
