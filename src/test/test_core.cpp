#include <gtest/gtest.h>

#include "../core.h"

using namespace ecvl;

TEST(Core, CreateEmptyImage) {
	Image img;
	EXPECT_EQ(img.dims_.size(), 0);
	EXPECT_EQ(img.strides_.size(), 0);
	EXPECT_EQ(img.data_, nullptr);
}

TEST(Core, CreateImageWithFiveDims) {

	Image img({ 1, 2, 3, 4, 5 }, DataType::uint8);
	EXPECT_EQ(img.dims_.size(), 5);
	for (int i = 0; i < img.dims_.size(); i++) {
		EXPECT_EQ(img.dims_[i], i + 1);
	}
	EXPECT_EQ(img.strides_.size(), 5);

}

