#include <gtest/gtest.h>

#include "ecvl/core.h"

using namespace ecvl;

TEST(Core, CreateEmptyImage) {
	Image img;
	EXPECT_EQ(img.dims_.size(), 0);
	EXPECT_EQ(img.strides_.size(), 0);
	EXPECT_EQ(img.data_, nullptr);
}

TEST(Core, CreateImageWithFiveDims) {

	Image img({ 1, 2, 3, 4, 5 }, DataType::uint8, "xyzoo", ColorType::none);
	EXPECT_EQ(img.dims_.size(), 5);
    int sdims = img.dims_.size();
	for (int i = 0; i < sdims; i++) {
		EXPECT_EQ(img.dims_[i], i + 1);
	}
	EXPECT_EQ(img.strides_.size(), 5);

}

