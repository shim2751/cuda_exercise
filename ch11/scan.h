#ifndef SCAN_H
#define SCAN_H

// Scan configuration
#define SECTION_SIZE 512    // Elements per section
#define SUBSEC_SIZE 16      // Elements per sub-section for coarsened scan

// Scan kernel types (following PDF chapter 11 progression)
typedef enum {
    SCAN_SEQUENTIAL = 0,        // Sequential scan (baseline)
    SCAN_KOGGE_STONE = 1,       // Kogge-Stone parallel scan
    SCAN_BRENT_KUNG = 2,        // Brent-Kung parallel scan
    SCAN_COARSENED = 3,         // Coarsened scan optimization
    SCAN_SEGMENTED = 4,          // Segmented scan for arbitrary-length inputs
    DOMINO_SCAN_SEGMENTED = 5 // Segmented scan with domino style
} scan_kernel_t;

// Unified launch function
void launch_scan(float* input_h, float* output_h, 
                 unsigned int length, scan_kernel_t kernel_type);

#endif