# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# def preprocess_data(temp_file, precip_file):
#     # Load datasets
#     temp_data = pd.read_csv(temp_file, skiprows=4)
#     precip_data = pd.read_csv(precip_file, skiprows=4)
    
#     # Rename columns for easier access
#     temp_data.columns = ['Year', 'Temp_Value', 'Temp_Anomaly', 'Temp_Uncertainty']
#     precip_data.columns = ['Year', 'Precipitation', 'Uncertainty']
    
#     # Merge the datasets on Year
#     merged_data = pd.merge(temp_data, precip_data, on='Year')
    
#     # Fill missing values
#     merged_data.fillna(method='ffill', inplace=True)
    
#     # Normalize the features
#     scaler = StandardScaler()
#     scaled_data = pd.DataFrame(scaler.fit_transform(merged_data[['Temp_Value', 'Precipitation']]), 
#                                columns=['Temp_Value', 'Precipitation'])
#     scaled_data['Year'] = merged_data['Year']
    
#     # Save the preprocessed data
#     scaled_data.to_csv('preprocessed_climate_data.csv', index=False)
#     print("Preprocessed data saved as preprocessed_climate_data.csv")

# if __name__ == "__main__":
#     preprocess_data('/Users/vishesh/Desktop/geo_project/data.csv', 'precipitation_data.csv')




import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(temp_file, precip_file):
    # Load datasets
    temp_data = pd.read_csv(temp_file, skiprows=4)
    precip_data = pd.read_csv(precip_file, skiprows=4)
    
    # Automatically rename columns based on the dataset (assumes first column is Year, and others are feature values)
    temp_data.columns = ['Year'] + list(temp_data.columns[1:])
    precip_data.columns = ['Year'] + list(precip_data.columns[1:])
    
    # Merge the datasets on 'Year' column
    merged_data = pd.merge(temp_data, precip_data, on='Year')
    
    # Fill missing values (forward fill method)
    merged_data.ffill(inplace=True)
    
    # Select all columns except 'Year' for normalization
    features = merged_data.drop(columns=['Year'])
    
    # Normalize all feature columns
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    # Add 'Year' column back to the scaled dataset
    scaled_data = pd.concat([merged_data[['Year']], scaled_features], axis=1)
    
    # Save the preprocessed data
    scaled_data.to_csv('preprocessed_climate_data.csv', index=False)
    print("Preprocessed data saved as preprocessed_climate_data.csv")

if __name__ == "__main__":
    preprocess_data('/Users/vishesh/Desktop/geo_project/data.csv', '/Users/vishesh/Desktop/geo_project/data-2.csv')

