#include <gtest/gtest.h>

#include "../core.h"

using namespace ecvl;
TEST(Core, CreateImage) {

	Image img1;
	EXPECT_EQ(img1.dims_.size(), 0);
	EXPECT_EQ(img1.strides_.size(), 0);
	EXPECT_EQ(img1.data_, nullptr);

	Image img2({1, 2, 3, 4, 5}, DataType::uint8);
	EXPECT_EQ(img2.dims_.size(), 5);
	for (int i = 0; i < img2.dims_.size(); i++) {
		EXPECT_EQ(img2.dims_[i], i + 1);
	}
	EXPECT_EQ(img2.strides_.size(), 5);

}

