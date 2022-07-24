
#include "matrix.h"

template <class T>
Matrix3D<T>::Matrix3D(size_t num_layers, size_t column_h, size_t row_w) {
    // set dimensions
    num_layers_ = num_layers;
    column_h_ = column_h;
    row_w_ = row_w;

    // data is always new allocated for 3D
    data_ = new T[num_layers_ * column_h_ * row_w_];

    // create an array of 2D matrices to return if indexed
    list_ = new Matrix2D<T>[num_layers];
    for (int i=0; i < num_layers_; i++) {
        // each matrix should reference piece of memory
        list_[i] = Matrix2D<T>(column_h_, row_w_, &(data_[i*column_h_*row_w_]));
    }
}
template <class T>
Matrix3D<T>::~Matrix3D() {
    // both were always new allocated
    delete[] list_;
    delete[] data_;
}

template <class T>
Matrix2D<T>::Matrix2D(size_t column_h, size_t row_w, T* data_loc) {
    // set dimensions
    column_h_ = column_h;
    row_w_ = row_w;

    // Allocate or reference data. Record if new allocated for later deletion.
    if (data_loc == NULL) {
        internal_ = false;
        data_ = new T[column_h_*row_w_];
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }

    // create an array of 2D matrices to return if indexed
    layer_ = new Matrix1D<T>[column_h_];
    for (int i=0; i < num_layers_; i++) {
        // each matrix should reference piece of memory
        layer_[i] = Matrix1D<T>(row_w_, &(data_[i*row_w_]));
    }
}
template <class T>
Matrix2D<T>::~Matrix2D() {
    // always new allocated
    delete[] layer_;
    // only delete if originally new allocated
    if (!internal_) delete[] data_;
}

template <class T>
Matrix1D<T>::Matrix1D(size_t row_w, T* data_loc) {
    // set dimensions
    row_w_ = row_w;

    // Allocate or reference data. Record if new allocated for later deletion.
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
    // only delete if originally new allocated
    if (!internal_) delete[] data_;
}