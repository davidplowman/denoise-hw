/*=============================================================================
Copyright (c) 2018 Raspberry Pi Trading Ltd.
=============================================================================*/

#ifndef _STAGE_HPP
#define _STAGE_HPP

class Image;

class Stage {
public:
  // Constructor and destructor. As this base class is stateless, there is nothing to do.
  Stage() { }
  virtual ~Stage() { }

  // Process a frame. The stage may assume dst != src and may process the pixels in any order.
  // Images will have sizes and types compatible with the stage's purpose and configuration.
  virtual void doFrame(Image & dst, Image const & src) = 0;

private:
  Stage(Stage const & rhs);             // copy constructor is not defined
  Stage & operator=(Stage const & rhs); // assignment operator is not defined
};

#endif
