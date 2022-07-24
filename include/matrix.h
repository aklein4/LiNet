#ifndef LAYER_H
#define LAYER_H

#include <assert.h>

// prototype data sub-structures
template <class T> class Matrix2D;
template <class T> class Matrix1D;

/* A 3D matrix, which is essentially a list of 2D matrixes. */
template <class T> 
class Matrix3D {
    public:
        Matrix3D(size_t num_layers, size_t column_h, size_t row_w);
        ~Matrix3D();

        Matrix2D<T>& operator [](size_t i) const {
            assert(i < num_layers_);
            return list_[i];
        };
    
        size_t num() {return num_layers_; };
        size_t height() {return column_h_; };
        size_t width() {return row_w_; };

    private:
        size_t num_layers_;
        size_t column_h_;
        size_t row_w_;
        Matrix2D<T>* list_;
        T* data_;
};

/* A 2D matrix, which is essentially a list of 1D rows. */
template <class T> 
class Matrix2D {
    public:
        Matrix2D(size_t column_h, size_t row_w, T* data_loc=NULL);
        ~Matrix2D();
        
        Matrix1D<T>& operator [](size_t i) const {
            assert(i < column_h_);
            return layer_[i];
        };

        size_t height() {return column_h_; };
        size_t width() {return row_w_; };

    private:
        size_t column_h_;
        size_t row_w_;
        Matrix1D<T>* layer_;
        bool internal_;
        T* data_;
};

/* A 1D Matrix row. */
template <class T>
class Matrix1D {
    public:
        Matrix1D(size_t row_w, T* data_loc=NULL);
        ~Matrix1D();

        T& operator [](size_t i) const {
            assert(i < row_w_);
            return data_[i];
        };

        size_t width() {return row_w_; };

    private:
        size_t row_w_;
        bool internal_;
        T* data_;
};

#endif