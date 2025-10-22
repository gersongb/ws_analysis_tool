#!/usr/bin/env python3
"""
Standalone script to compare two d1_processed CSV exports and generate a delta file.

Matches rows based on: frh, rrh, roll, yaw
Calculates deltas for all numeric columns.

Usage:
    python compare_runs.py <csv_file_1> <csv_file_2> [output_delta_csv]

Example:
    python compare_runs.py "Run0003_d1_processed.csv" "Run0004_d1_processed.csv"
    python compare_runs.py "Run0003_d1_processed.csv" "Run0004_d1_processed.csv" "delta.csv"
"""

import sys
import os
import csv
from collections import defaultdict


def read_csv_with_headers(csv_path):
    """Read CSV file and return headers and rows as dictionaries."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return None, None
    
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
        
        print(f"[OK] Loaded {csv_path}")
        print(f"  Rows: {len(rows)}")
        print(f"  Columns: {len(headers)}")
        
        return headers, rows
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None


def create_match_key(row, key_columns):
    """Create a tuple key from specified columns for matching rows."""
    try:
        # Convert to float for proper comparison (handles string numbers)
        values = []
        for col in key_columns:
            val = row.get(col, '').strip()
            if val == '' or val == 'nan':
                values.append(None)
            else:
                try:
                    values.append(float(val))
                except ValueError:
                    values.append(val)
        return tuple(values)
    except Exception:
        return None


def is_numeric(value):
    """Check if a value can be converted to float."""
    if value is None or value == '' or value == 'nan':
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def calculate_delta(val1, val2):
    """Calculate delta between two values. Returns val2 - val1."""
    try:
        if not is_numeric(val1) or not is_numeric(val2):
            return ''
        return float(val2) - float(val1)
    except Exception:
        return ''


def compare_runs(csv1_path, csv2_path, output_csv=None):
    """
    Compare two CSV files and generate a delta file.
    
    Args:
        csv1_path: Path to first CSV (baseline)
        csv2_path: Path to second CSV (comparison)
        output_csv: Optional output path for delta CSV
    
    Returns:
        True if successful, False otherwise
    """
    # Read both CSV files
    headers1, rows1 = read_csv_with_headers(csv1_path)
    headers2, rows2 = read_csv_with_headers(csv2_path)
    
    if headers1 is None or headers2 is None:
        return False
    
    # Check that headers match
    if set(headers1) != set(headers2):
        print("Warning: Column headers don't match between files")
        print(f"  CSV1 columns: {len(headers1)}")
        print(f"  CSV2 columns: {len(headers2)}")
        # Use intersection of headers
        common_headers = [h for h in headers1 if h in headers2]
        print(f"  Using common columns: {len(common_headers)}")
    else:
        common_headers = headers1
    
    # Define matching key columns
    key_columns = ['frh', 'rrh', 'roll', 'yaw']
    
    # Verify key columns exist
    missing_keys = [k for k in key_columns if k not in common_headers]
    if missing_keys:
        print(f"Error: Missing key columns: {missing_keys}")
        return False
    
    print(f"\nMatching rows based on: {', '.join(key_columns)}")
    
    # Index CSV1 by match key
    csv1_index = {}
    for row in rows1:
        key = create_match_key(row, key_columns)
        if key and key not in csv1_index:
            csv1_index[key] = row
    
    print(f"\nCSV1 unique combinations: {len(csv1_index)}")
    
    # Find matching rows and calculate deltas
    matched_rows = []
    unmatched_count = 0
    
    for row2 in rows2:
        key = create_match_key(row2, key_columns)
        if key and key in csv1_index:
            row1 = csv1_index[key]
            
            # Create delta row
            delta_row = {}
            
            # Keep key columns from row2
            for col in key_columns:
                delta_row[col] = row2.get(col, '')
            
            # Keep setpoint columns
            if 'setpoint' in common_headers:
                delta_row['setpoint_csv1'] = row1.get('setpoint', '')
                delta_row['setpoint_csv2'] = row2.get('setpoint', '')
            
            # Calculate deltas for other columns
            for col in common_headers:
                if col in key_columns or col == 'setpoint':
                    continue
                
                val1 = row1.get(col, '')
                val2 = row2.get(col, '')
                
                if is_numeric(val1) and is_numeric(val2):
                    delta = calculate_delta(val1, val2)
                    delta_row[f'delta_{col}'] = f"{delta:.6f}" if delta != '' else ''
                    delta_row[f'{col}_csv1'] = val1
                    delta_row[f'{col}_csv2'] = val2
                else:
                    # Keep non-numeric values as-is
                    if val1 == val2:
                        delta_row[col] = val1
                    else:
                        delta_row[f'{col}_csv1'] = val1
                        delta_row[f'{col}_csv2'] = val2
            
            matched_rows.append(delta_row)
        else:
            unmatched_count += 1
    
    print(f"Matched rows: {len(matched_rows)}")
    print(f"Unmatched rows in CSV2: {unmatched_count}")
    
    if len(matched_rows) == 0:
        print("Error: No matching rows found!")
        return False
    
    # Generate output filename if not specified
    if output_csv is None:
        csv1_name = os.path.splitext(os.path.basename(csv1_path))[0]
        csv2_name = os.path.splitext(os.path.basename(csv2_path))[0]
        output_dir = os.path.dirname(csv1_path)
        output_csv = os.path.join(output_dir, f"delta_{csv1_name}_vs_{csv2_name}.csv")
    
    # Write delta CSV
    try:
        # Build header list: key columns, setpoints, then delta columns
        delta_headers = key_columns.copy()
        
        if 'setpoint' in common_headers:
            delta_headers.extend(['setpoint_csv1', 'setpoint_csv2'])
        
        # Add columns in order: delta, csv1, csv2
        numeric_cols = []
        for col in common_headers:
            if col in key_columns or col == 'setpoint':
                continue
            if f'delta_{col}' in matched_rows[0]:
                numeric_cols.append(col)
        
        for col in numeric_cols:
            delta_headers.append(f'delta_{col}')
            delta_headers.append(f'{col}_csv1')
            delta_headers.append(f'{col}_csv2')
        
        # Write CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=delta_headers, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(matched_rows)
        
        print(f"\n[OK] Delta CSV saved to: {output_csv}")
        print(f"  Matched rows: {len(matched_rows)}")
        print(f"  Columns: {len(delta_headers)}")
        
        return True
    
    except Exception as e:
        print(f"Error writing delta CSV: {e}")
        return False


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    csv1_path = sys.argv[1]
    csv2_path = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("="*60)
    print("Run Comparison Tool")
    print("="*60)
    print(f"CSV 1 (baseline): {os.path.basename(csv1_path)}")
    print(f"CSV 2 (compare):  {os.path.basename(csv2_path)}")
    print("="*60 + "\n")
    
    success = compare_runs(csv1_path, csv2_path, output_csv)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
