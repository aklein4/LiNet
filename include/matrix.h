#ifndef LAYER_H
#define LAYER_H

#include <assert.h>

// prototype data sub-structures
template <class T> class Matrix2D;
template <class T> class Matrix1D;

/* A 3D matrix, which is essentially a list of 2D matrixes.
 * Data is stored contiguously in memory, and indexing returns a 
 * 2D matrix whose data is a section of that memory.
 */
template <class T> 
class Matrix3D {
    public:
        /* Create a 3D matrix of the specified size.
         * If data will be new allocated and deleted upon deconstruction.
         * \param[in] num_layers Number of 2D matrix layers
         * \param[in] column_h Height of a child 2D matrix's column
         * \param[in] row_w Width of a child 2D matrix's row
         */
        Matrix3D(size_t num_layers, size_t column_h, size_t row_w);
        /* Data is deleted upon deconstruction, and any child matrices
         * will index to invalid data. */
        ~Matrix3D();

        /* Return a 2D matrix which references a piece of the contiguous memory.
         */
        Matrix2D<T>& operator [](size_t i) const {
            assert(i < num_layers_);
            return list_[i];
        };
    
        /* \return Number of 2D sub-matrices */
        size_t num() {return num_layers_; };
        /* \return Height of column */
        size_t height() {return column_h_; };
        /* \return Width of row */
        size_t width() {return row_w_; };

    private:
        // dimensions
        size_t num_layers_;
        size_t column_h_;
        size_t row_w_;
        // pointer to memory chunk containing data
        T* data_;
        // an array to hold sub-matrices
        Matrix2D<T>* list_;
};

/* A 2D matrix, which is essentially a list of 1D matrixes.
 * Data is stored contiguously in memory, and indexing returns a 
 * 1D matrix whose data is a section of that memory.
 */
template <class T> 
class Matrix2D {
    public:
        /* Create a 2D matrix of the specified size.
         * If data_loc is not NULL, then the data will be set to 
         * that location and will not be deleted upon deconstuction.
         * \param[in] column_h Height of a child 2D matrix's column
         * \param[in] row_w Width of a child 2D matrix's row
         * \param[in] data_loc Pointer to the data to reference. Default NULL means new allocation.
         */
        Matrix2D(size_t column_h, size_t row_w, T* data_loc=NULL);
        /* Deconstuction only deletes data if it was created in the constructor.
         * If that happens, then child 1D matrixes will index to invalid memory.*/
        ~Matrix2D();
        
        /* Return a 1D matrix which references a piece of the contiguous memory.
         */
        Matrix1D<T>& operator [](size_t i) const {
            assert(i < column_h_);
            return layer_[i];
        };

        /* \return Height of column */
        size_t height() {return column_h_; };
        /* \return Width of row */
        size_t width() {return row_w_; };

    private:
        // dimensions
        size_t column_h_;
        size_t row_w_;
        // pointer to memory chunk containing data
        T* data_;
        // whether this is a sub-matrix, if not then data must be deleted
        bool internal_;
        // an array to hold sub-matrices
        Matrix1D<T>* layer_;
};

/* A 1D matrix, whose data is stored contigously in memory.
 * Indexing returns a reference to an entry in that memory.
 */
template <class T>
class Matrix1D {
    public:
        /* Create a 1D matrix of the specified size.
         * If data_loc is not NULL, then the data will be set to 
         * that location and will not be deleted upon deconstuction.
         * \param[in] row_w Width of a child 2D matrix's row
         * \param[in] data_loc Pointer to the data to reference. Default NULL means new allocation.
         */
        Matrix1D(size_t row_w, T* data_loc=NULL);
        /* Deconstuction only deletes data if it was created in the constructor */
        ~Matrix1D();

        /* Return a reference to a piece of the contiguous memory.
         */
        T& operator [](size_t i) const {
            assert(i < row_w_);
            return data_[i];
        };

        /* \return Width of row */
        size_t width() {return row_w_; };

    private:
        // dimensions
        size_t row_w_;
        // pointer to memory chunk containing data
        T* data_;
        // whether this is a sub-matrix, if not then data must be deleted
        bool internal_;
};

#endif