
#include "matrix.h"

template <class T>
Matrix3D<T>::Matrix3D(size_t num_layers, size_t column_h, size_t row_w) {
    num_layers_ = num_layers;
    column_h_ = column_h;
    row_w_ = row_w;

    data_ = new T[num_layers_ * column_h_ * row_w_];

    list_ = new Matrix2D<T>[num_layers];
    for (int i=0; i < num_layers_; i++) {
        list_[i] = Matrix2D<T>(column_h_, row_w_, &(data_[i*column_h_*row_w_]));
    }
}
template <class T>
Matrix3D<T>::~Matrix3D() {
    delete[] list_;
    delete[] data_;
}

template <class T>
Matrix2D<T>::Matrix2D(size_t column_h, size_t row_w, T* data_loc) {
    column_h_ = column_h;
    row_w_ = row_w;

    if (data_loc == NULL) {
        internal_ = false;
        data_ = new T[column_h_*row_w_];
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }

    layer_ = new Matrix1D<T>[column_h_];
    for (int i=0; i < num_layers_; i++) {
        layer_[i] = Matrix1D<T>(row_w_, &(data_[i*row_w_]));
    }
}
template <class T>
Matrix2D<T>::~Matrix2D() {
    delete[] layer_;
    if (!internal_) delete[] data_;
}

template <class T>
Matrix1D<T>::Matrix1D(size_t row_w, T* data_loc) {
    row_w_ = row_w;

    if (data_loc == NULL) {
        internal_ = false;
        data_ = new T[row_w_];
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }
}
template <class T>
Matrix1D<T>::~Matrix1D() {
    if (!internal_) delete[] data_;
}