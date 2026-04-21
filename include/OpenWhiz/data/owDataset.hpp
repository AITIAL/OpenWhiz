/*
 * owDataset.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <random>
#include <chrono>
#include "../core/owTensor.hpp"

/**
 * @file owDataset.hpp
 * @brief Data management and preprocessing utilities for OpenWhiz.
 */

namespace ow {

/**
 * @enum DataType
 * @brief Supported data types for dataset columns.
 */
enum class DataType { 
    Numeric,   ///< Continuous or discrete numerical values.
    Datetime,  ///< Date and time strings (to be encoded).
    Text       ///< Categorical or raw text data.
};

/**
 * @enum Ordering
 * @brief Categorical ordering strategies.
 */
enum class Ordering { 
    Standard,     ///< No special ordering.
    Categorical,  ///< Unordered categories (Nominal).
    Ordered       ///< Categories with a specific sequence (Ordinal).
};

/**
 * @enum ColumnUsage
 * @brief Usage status of a column for training and calculation.
 */
enum class ColumnUsage {
    USED,     ///< Column is used for training and calculation.
    UNUSED,   ///< Column is loaded but explicitly excluded from training and calculation.
    ORDERING  ///< Column is used as a sequence marker (e.g., Step No, Time), not for training.
};

/**
 * @enum SampleType
 * @brief Categorization for dataset splitting.
 */
enum class SampleType { 
    Training,    ///< Data used for model parameter updates.
    Validation,  ///< Data used for hyperparameter tuning and early stopping.
    Test         ///< Unseen data for final performance evaluation.
};

/**
 * @enum ImputationStrategy
 * @brief Strategies for handling missing data (NaN/Empty).
 */
enum class ImputationStrategy { 
    Mean,         ///< Replace with the column mean.
    Zero,         ///< Replace with 0.0.
    ForwardFill   ///< Replace with the previous valid value.
};

/**
 * @struct ColumnInfo
 * @brief Metadata for a single dataset column.
 */
struct ColumnInfo {
    std::string name;                          ///< Name of the column (from CSV header).
    DataType type;                             ///< Interpreted data type.
    Ordering ordering;                         ///< Categorical ordering type.
    ColumnUsage usage = ColumnUsage::USED;     ///< Usage status for training.
    std::map<std::string, float> category_map; ///< Mapping from category string to float ID.
    std::vector<std::string> reverse_category_map; ///< Mapping from float ID back to category string.
    float min = 0.0f;                          ///< Minimum value observed in the data.
    float max = 1.0f;                          ///< Maximum value observed in the data.
};

/**
 * @class owDataset
 * @brief Core class for data loading, preprocessing, and management.
 */
class owDataset {
public:
    owDataset() : m_targetVariableNum(1), m_autoNormalizeEnabled(false) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        m_rng.seed(static_cast<unsigned int>(seed));
    }
    ~owDataset() = default;

    /**
     * @brief Loads and parses a CSV file.
     * 
     * Filtering: Columns ending in "ID" (case-insensitive) are not loaded.
     * Categorical: Automatically detects text columns and applies label encoding.
     * Delimiter: Automatically detected if not explicitly set.
     */
    bool loadFromCSV(const std::string& filepath, bool has_header = true, bool autoNormalize = false) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        std::string line;

        // Automatic Delimiter Detection
        if (std::getline(file, line)) {
            char candidates[] = {';', '|', '\t', ','};
            char best_d = m_delimiter;
            int max_c = 0;
            
            for (char cnd : candidates) {
                int count = (int)std::count(line.begin(), line.end(), cnd);
                // If we find a candidate with more occurrences, it's the new best.
                // The order of 'candidates' ensures priority in case of ties.
                if (count > max_c) {
                    max_c = count;
                    best_d = cnd;
                }
            }
            m_delimiter = best_d;
            
            // Reset file pointer to beginning
            file.clear();
            file.seekg(0);
        }

        std::vector<int> col_indices; // Indices of columns to keep
        
        auto isID = [](const std::string& name) {
            if (name.length() < 2) return false;
            std::string suffix = name.substr(name.length() - 2);
            for (char &c : suffix) c = std::toupper(c);
            return suffix == "ID";
        };

        if (has_header && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string col;
            int idx = 0;
            while (std::getline(ss, col, m_delimiter)) {
                if (!isID(col)) {
                    m_columns.push_back({col, DataType::Numeric, Ordering::Standard, ColumnUsage::USED, {}, {}, 0.0f, 1.0f});
                    col_indices.push_back(idx);
                }
                idx++;
            }
        }

        std::vector<std::vector<std::string>> raw_data;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            std::vector<std::string> full_row;
            while (std::getline(ss, val, m_delimiter)) full_row.push_back(val);
            
            if (m_columns.empty()) {
                for (size_t i = 0; i < full_row.size(); ++i) {
                    std::string name = "col_" + std::to_string(i);
                    if (!isID(name)) {
                        m_columns.push_back({name, DataType::Numeric, Ordering::Standard, ColumnUsage::USED, {}, {}, 0.0f, 1.0f});
                        col_indices.push_back((int)i);
                    }
                }
            }

            std::vector<std::string> filtered_row;
            for (int idx : col_indices) {
                if ((size_t)idx < full_row.size()) filtered_row.push_back(full_row[idx]);
                else filtered_row.push_back("0");
            }
            raw_data.push_back(filtered_row);
        }

        if (raw_data.empty()) return false;
        size_t rows = raw_data.size();
        size_t cols = m_columns.size();

        // Detect Data Types and Categorical Mapping
        for (size_t c = 0; c < cols; ++c) {
            bool all_numeric = true;
            for (size_t r = 0; r < rows; ++r) {
                const std::string& val = raw_data[r][c];
                if (val.empty()) continue;
                try {
                    std::size_t pos;
                    std::stof(val, &pos);
                    if (pos != val.length()) all_numeric = false;
                } catch (...) {
                    all_numeric = false;
                }
                if (!all_numeric) break;
            }

            if (!all_numeric) {
                m_columns[c].type = DataType::Text;
                float next_id = 0.0f;
                for (size_t r = 0; r < rows; ++r) {
                    const std::string& val = raw_data[r][c];
                    if (m_columns[c].category_map.find(val) == m_columns[c].category_map.end()) {
                        m_columns[c].category_map[val] = next_id++;
                        m_columns[c].reverse_category_map.push_back(val);
                    }
                }
            }
        }

        m_fullData = owTensor<float, 2>(rows, cols);
        for (size_t c = 0; c < cols; ++c) {
            for (size_t r = 0; r < rows; ++r) m_fullData(r, c) = parseValue(raw_data[r][c], m_columns[c]);
        }

        // Always calculate statistics (min/max) even if not normalizing in-place
        calculateStatistics();

        m_sampleTypes.assign(rows, SampleType::Training);
        shuffleSampleTypes();
        m_autoNormalizeEnabled = autoNormalize;
        if (m_autoNormalizeEnabled) normalizeData();
        return true;
    }

    void setAutoNormalizeEnabled(bool enable) { m_autoNormalizeEnabled = enable; }
    void setTargetVariableNum(int num) { m_targetVariableNum = num; }
    int getTargetVariableNum() const { return m_targetVariableNum; }

    /**
     * @brief Calculates min/max statistics for each column in the dataset.
     */
    void calculateStatistics() {
        if (m_fullData.size() == 0) return;
        size_t rows = m_fullData.shape()[0];
        size_t cols = m_fullData.shape()[1];
        for (size_t c = 0; c < cols; ++c) {
            float minVal = 1e30f, maxVal = -1e30f;
            for (size_t r = 0; r < rows; ++r) {
                minVal = std::min(minVal, m_fullData(r, c));
                maxVal = std::max(maxVal, m_fullData(r, c));
            }
            m_columns[c].min = minVal;
            m_columns[c].max = maxVal;
        }
    }

    /**
     * @brief Returns the string label for a numeric category ID.
     */
    std::string getLabelName(int actualColIdx, float value) const {
        if (actualColIdx < 0 || (size_t)actualColIdx >= m_columns.size()) return "";
        const auto& info = m_columns[actualColIdx];
        if (info.type != DataType::Text) return std::to_string(value);
        
        int id = (int)std::round(value);
        if (id >= 0 && (size_t)id < info.reverse_category_map.size()) {
            return info.reverse_category_map[id];
        }
        return std::to_string(value);
    }

    /**
     * @brief Gets the actual column index in the full dataset for a target variable.
     */
    int getTargetColumnIndex(int targetVarIdx = 0) const {
        int inputColsCount = (int)m_columns.size() - m_targetVariableNum;
        return inputColsCount + targetVarIdx;
    }

    /**
     * @brief Marks a column by name with a specific usage.
     */
    void setColumnUsage(const std::string& name, ColumnUsage usage) {
        for (auto& col : m_columns) {
            if (col.name == name) {
                col.usage = usage;
                return;
            }
        }
    }

    /**
     * @brief Returns indices of columns that should be used for training/calculation.
     */
    std::vector<int> getUsedColumnIndices(bool includeTarget = false) const {
        std::vector<int> indices;
        int inputCols = (int)m_columns.size() - m_targetVariableNum;
        for (int i = 0; i < inputCols; ++i) {
            if (m_columns[i].usage == ColumnUsage::USED) indices.push_back(i);
        }
        if (includeTarget) {
            for (int i = inputCols; i < (int)m_columns.size(); ++i) indices.push_back(i);
        }
        return indices;
    }

    int getInputVariableNum() const { 
        int count = 0;
        int inputCols = (int)m_columns.size() - m_targetVariableNum;
        for (int i = 0; i < inputCols; ++i) if (m_columns[i].usage == ColumnUsage::USED) count++;
        return count;
    }

    size_t getSampleNum() const { return m_fullData.shape()[0]; }

    owTensor<float, 2> getData() const { return m_fullData; }

    /**
     * @brief Normalizes the entire dataset in-place using Min-Max scaling.
     * After this call, m_fullData will contain values in the [0, 1] range.
     */
    void normalizeData() {
        if (m_fullData.size() == 0) return;
        calculateStatistics();
        size_t rows = m_fullData.shape()[0];
        size_t cols = m_fullData.shape()[1];
        for (size_t c = 0; c < cols; ++c) {
            float minVal = m_columns[c].min;
            float maxVal = m_columns[c].max;
            float range = maxVal - minVal;
            if (range == 0.0f) range = 1.0f;
            
            for (size_t r = 0; r < rows; ++r) {
                m_fullData(r, c) = (m_fullData(r, c) - minVal) / range;
            }
        }
        m_autoNormalizeEnabled = false; // Scaled in-place, no need for on-the-fly scaling
    }

    /**
     * @brief Scales a prediction or value back to its original range.
     */
    void inverseNormalize(owTensor<float, 2>& data, int targetVarIdx = 0) {
        int actualColIdx = getTargetColumnIndex(targetVarIdx);
        float minV = m_columns[actualColIdx].min;
        float maxV = m_columns[actualColIdx].max;
        float range = maxV - minV;
        if (range == 0.0f) range = 1.0f;
        
        for (size_t i = 0; i < data.shape()[0]; ++i) {
            for (size_t j = 0; j < data.shape()[1]; ++j) {
                data(i, j) = data(i, j) * range + minV;
            }
        }
    }

    /**
     * @brief Transforms the dataset into a windowed format for time-series forecasting.
     * This expands the features by prepending historical values to each sample.
     */
    void prepareForecastData(int windowSize, int dilation = 1) {
        if (m_fullData.size() == 0 || windowSize <= 0) return;
        size_t originalRows = m_fullData.shape()[0];
        size_t originalCols = m_fullData.shape()[1];
        int inputCols = (int)originalCols - m_targetVariableNum;
        
        size_t offset = (size_t)windowSize * (size_t)dilation;
        if (originalRows <= offset) return;

        size_t newRows = originalRows - offset;
        size_t newCols = (size_t)windowSize + originalCols;
        
        owTensor<float, 2> newData(newRows, newCols);
        std::vector<SampleType> newSampleTypes(newRows);

        for (size_t i = 0; i < newRows; ++i) {
            size_t actualIdx = i + offset;
            
            // 1. Fill History (using the first target variable as default reference)
            for (int w = 0; w < windowSize; ++w) {
                size_t lookback = (size_t)(windowSize - w) * (size_t)dilation;
                newData(i, (size_t)w) = m_fullData(actualIdx - lookback, (size_t)inputCols);
            }
            
            // 2. Fill current row data
            for (size_t j = 0; j < originalCols; ++j) {
                newData(i, (size_t)windowSize + j) = m_fullData(actualIdx, j);
            }
            
            newSampleTypes[i] = m_sampleTypes[actualIdx];
        }

        m_fullData = newData;
        m_sampleTypes = newSampleTypes;

        // Update column metadata
        std::vector<ColumnInfo> newColumns;
        for (int w = 0; w < windowSize; ++w) {
            newColumns.push_back({"History_" + std::to_string(windowSize - w), DataType::Numeric, Ordering::Standard, ColumnUsage::USED});
        }
        for (const auto& col : m_columns) newColumns.push_back(col);
        m_columns = newColumns;
    }

    owTensor<float, 2> getLastSample() const {
        std::vector<int> indices = getUsedColumnIndices(false);
        if (m_fullData.shape()[0] == 0 || indices.empty()) return owTensor<float, 2>(0, 0);
        owTensor<float, 2> res(1, indices.size());
        size_t lastRow = m_fullData.shape()[0] - 1;
        for (size_t j = 0; j < indices.size(); ++j) res(0, j) = m_fullData(lastRow, (size_t)indices[j]);
        return res;
    }

    std::pair<float, float> getNormalizationParams(int usedColIdx) const {
        std::vector<int> indices = getUsedColumnIndices(true);
        if (usedColIdx < 0 || (size_t)usedColIdx >= indices.size()) return {0.0f, 1.0f};
        int actualIdx = indices[usedColIdx];
        return {m_columns[actualIdx].min, m_columns[actualIdx].max};
    }

    std::pair<float, float> getNormalizationParams(const std::string& name) const {
        for (const auto& col : m_columns) {
            if (col.name == name) return {col.min, col.max};
        }
        return {0.0f, 1.0f};
    }

    owTensor<float, 2> getRowsAndColsFiltered(SampleType targetType, bool isInput) const {
        std::vector<int> colIndices;
        if (isInput) {
            colIndices = getUsedColumnIndices(false);
        } else {
            int inputCols = (int)m_columns.size() - m_targetVariableNum;
            for (int i = inputCols; i < (int)m_columns.size(); ++i) colIndices.push_back(i);
        }

        size_t rows = 0;
        for (auto t : m_sampleTypes) if (t == targetType) rows++;
        if (rows == 0) return owTensor<float, 2>(0, colIndices.size());

        owTensor<float, 2> res(rows, colIndices.size());
        size_t curr = 0;
        for (size_t i = 0; i < m_sampleTypes.size(); ++i) {
            if (m_sampleTypes[i] == targetType) {
                for (size_t j = 0; j < colIndices.size(); ++j) {
                    int colIdx = colIndices[j];
                    float val = m_fullData(i, (size_t)colIdx);
                    
                    // Apply scaling ON-THE-FLY if enabled
                    if (m_autoNormalizeEnabled && m_columns[colIdx].usage == ColumnUsage::USED) {
                        float minV = m_columns[colIdx].min;
                        float maxV = m_columns[colIdx].max;
                        float range = maxV - minV;
                        if (range == 0) range = 1.0f;
                        val = (val - minV) / range;
                    }
                    
                    res(curr, j) = val;
                }
                curr++;
            }
        }
        return res;
    }

    owTensor<float, 2> getTrainInput() const { return getRowsAndColsFiltered(SampleType::Training, true); }
    owTensor<float, 2> getTrainTarget() const { return getRowsAndColsFiltered(SampleType::Training, false); }
    owTensor<float, 2> getValInput() const { return getRowsAndColsFiltered(SampleType::Validation, true); }
    owTensor<float, 2> getValTarget() const { return getRowsAndColsFiltered(SampleType::Validation, false); }
    owTensor<float, 2> getTestInput() const { return getRowsAndColsFiltered(SampleType::Test, true); }
    owTensor<float, 2> getTestTarget() const { return getRowsAndColsFiltered(SampleType::Test, false); }

    void setRatios(float train, float val, float test, bool shuffle = true) {
        m_trainRatio = train; m_valRatio = val; m_testRatio = test;
        shuffleSampleTypes(shuffle);
    }

    void setDelimiter(char d) { m_delimiter = d; }
    char getDelimiter() const { return m_delimiter; }

    void shuffleSampleTypes(bool shuffle = true) {
        if (m_sampleTypes.empty()) return;
        size_t rows = m_sampleTypes.size();
        size_t trainCount = (size_t)(rows * m_trainRatio);
        size_t valCount = (size_t)(rows * m_valRatio);
        
        std::vector<SampleType> newTypes(rows);
        for (size_t i = 0; i < rows; ++i) {
            if (i < trainCount) newTypes[i] = SampleType::Training;
            else if (i < trainCount + valCount) newTypes[i] = SampleType::Validation;
            else newTypes[i] = SampleType::Test;
        }
        
        if (shuffle) {
            std::shuffle(newTypes.begin(), newTypes.end(), m_rng);
        }
        m_sampleTypes = newTypes;
    }

    std::string getSampleTypeString(size_t index) const {
        if (index >= m_sampleTypes.size()) return "Unknown";
        if (m_sampleTypes[index] == SampleType::Training) return "Training";
        if (m_sampleTypes[index] == SampleType::Validation) return "Validation";
        return "Testing";
    }

private:
    owTensor<float, 2> m_fullData;
    std::vector<ColumnInfo> m_columns;
    std::vector<SampleType> m_sampleTypes;
    int m_targetVariableNum = 1;
    float m_trainRatio = 0.6f, m_valRatio = 0.2f, m_testRatio = 0.2f;
    char m_delimiter = ';'; // Default changed to semicolon
    bool m_autoNormalizeEnabled = false;
    std::mt19937 m_rng;

    float parseValue(const std::string& val, ColumnInfo& info) {
        if (info.type == DataType::Text) {
            auto it = info.category_map.find(val);
            if (it != info.category_map.end()) return it->second;
            return 0.0f;
        }
        try { return std::stof(val); } catch (...) { return 0.0f; }
    }
};

} // namespace ow
