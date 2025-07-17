
#ifndef CH3_H
#define CH3_H

#define CHANNEL 3

void color_to_grayscale(unsigned char* Pin, unsigned char* Pout, int width, int height);
void image_blur(unsigned char* Pin, unsigned char* Pout, int width, int height, int radius);

#endif
