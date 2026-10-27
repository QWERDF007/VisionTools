#include <exception>
#include <iostream>

void testAdaptiveBinaryTool();
void testRoiProjectionTool();
void testImageCropTool();
void testDetectionFilterTool();
void testImagePreprocessTool();
void testReadImageTool();

int main()
{
    try
    {
        testAdaptiveBinaryTool();
        testRoiProjectionTool();
        testImageCropTool();
        testDetectionFilterTool();
        testImagePreprocessTool();
        testReadImageTool();
        std::cout << "\n[ALL TESTS PASSED] All preprocess contract tests passed successfully.\n" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FATAL] Unhandled exception: " << error.what() << std::endl;
        return 1;
    }
}
