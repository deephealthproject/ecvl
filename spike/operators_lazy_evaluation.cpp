#include <iostream>

struct op;

struct a {
    int i;
    bool empty = false;

    a(int i_) : i{ i_ } { std::cout << "constructor\n"; }
    a(const a& other) : i{ other.i } { std::cout << "copy constructor\n"; }
    a(a&& other) : i{ other.i } { other.empty = true; std::cout << "move constructor\n"; }
    friend void swap(a& lhs, a& rhs) {
        using std::swap;
        swap(lhs.i, rhs.i);
    }
    a& operator=(a other) {
        std::cout << "assignment\n";
        swap(*this, other);
        return *this;
    }
    ~a() { std::cout << "destructor (empty=" << empty << ")\n"; }

    a& operator=(const op& other);

    a& operator+=(const a& rhs) {
        i += rhs.i;
        return *this;
    }

    friend op operator+(const a& lhs, const a& rhs);
    friend op operator-(const a& lhs, const a& rhs);
    friend op operator*(const a& lhs, const a& rhs);
    friend op operator/(const a& lhs, const a& rhs);
    friend op operator-(const a& lhs);

    int get() const { return i; }
};

struct op {
    const a* obj_ = nullptr;
    const op* lhs_ = nullptr;
    const op* rhs_ = nullptr;
    char oper_ = 0;

    op(const a& obj) : obj_{ &obj } {}
    op(const op& lhs, char oper) : lhs_{ new op(lhs) }, oper_{ oper } {}
    op(const op& lhs, const op& rhs, char oper) : lhs_{ new op(lhs) }, rhs_{ new op(rhs) }, oper_{ oper } {}

    friend op operator+(const op& lhs, const op& rhs) { return op(lhs, rhs, '+'); }
    friend op operator-(const op& lhs, const op& rhs) { return op(lhs, rhs, '-'); }
    friend op operator*(const op& lhs, const op& rhs) { return op(lhs, rhs, '*'); }
    friend op operator/(const op& lhs, const op& rhs) { return op(lhs, rhs, '/'); }

    friend op operator-(const op& lhs) { return op(lhs, 'n'); }

    int get() const {
        switch (oper_) {
        case 0: return obj_->get();
        case 'n': return -lhs_->get();
        case '+': return lhs_->get() + rhs_->get();
        case '-': return lhs_->get() - rhs_->get();
        case '*': return lhs_->get() * rhs_->get();
        case '/': return lhs_->get() / rhs_->get();
        default: throw;
        }
    }
};

op operator+(const a& lhs, const a& rhs) { return op(lhs, rhs, '+'); }
op operator-(const a& lhs, const a& rhs) { return op(lhs, rhs, '-'); }
op operator*(const a& lhs, const a& rhs) { return op(lhs, rhs, '*'); }
op operator/(const a& lhs, const a& rhs) { return op(lhs, rhs, '/'); }
op operator-(const a& lhs) { return op(lhs, 'n'); }

a& a::operator=(const op& other) {
    i = other.get();
    return *this;
}

int main(void)
{
    
    a x(3), y(7), z(1);

    auto k = x + y * (-z);

    std::cout << k.get() << "\n";

    return 0;
    
}