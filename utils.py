# plot_formatter.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def format_plot(fig, title, xaxis_title, yaxis_title, x_tick_format='.2f'):
    """
    This function formats a Plotly figure with customized settings.
    
    Parameters:
    - fig: Plotly figure object to be formatted
    - title: The title of the plot
    - xaxis_title: The title for the X axis
    - yaxis_title: The title for the Y axis
    - x_tick_format: Format for the x-axis ticks (default is '.2f')
    
    Returns:
    - fig: The formatted Plotly figure object
    """
    
    fig.update_layout(
        title=title,
        title_x=0.5,  # Center title
        title_font=dict(size=20, family='Arial', color='black'),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(
            tickformat=x_tick_format,
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.3)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 0, 0, 0.3)'
        ),
        plot_bgcolor='rgb(255, 255, 255)',  # white background
        paper_bgcolor='rgb(255, 255, 255)',  # Paper background
        font=dict(color='black'),  # black font for readability
        bargap=0.1  # Bar gap
    )
    
    return fig



def plot_beautiful_corr_heatmap(df, numeric_columns):
    """
    This function plots a beautiful correlation matrix heatmap with improved aesthetics.
    Arguments:
    - df: The dataframe with the data.
    - numeric_columns: A list of numeric columns to include in the correlation matrix.
    """
    # Filter the columns to exclude 'FlownYear' and add 'PricePerWeight' if needed
    main_numeric_columns = list(numeric_columns)
    main_numeric_columns.append('PricePerWeight')
    
    # Calculate the correlation matrix
    corr_matrix = df.loc[:, main_numeric_columns].corr()

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlBu',  # A more visually intuitive color scale
        colorbar=dict(title="Correlation", tickvals=[-1, 0, 1], ticktext=["-1", "0", "1"]),
        zmin=-1, zmax=1  # Fixed color range for better comparability
    ))

    # Update layout for better aesthetics
    fig.update_layout(
        title='Correlation Matrix',
        title_x=0.5,  # Center the title
        title_font=dict(size=20, color='black'),
        xaxis=dict(
            title='Features',
            tickangle=45,  # Rotate x-axis labels for better readability
            tickfont=dict(size=12),
            showgrid=False  # Remove x-axis grid lines
        ),
        yaxis=dict(
            title='Features',
            tickangle=45,  # Rotate y-axis labels for better readability
            tickfont=dict(size=12),
            showgrid=False  # Remove y-axis grid lines
        ),
        template="plotly_white",  # Clean background
        height=600,  # Adjust the figure height
        width=600,   # Adjust the figure width
    )
    
    # Add annotations (correlation values in each cell)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                go.layout.Annotation(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.columns[i],
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=10, color="black"),
                    align="center"
                )
            )
    
    fig.show()



def plot_grouped_bar_chart(df, x_col, y_col1, y_col2, title):
    fig = go.Figure(data=[
        go.Bar(
            name=y_col1, 
            x=df[x_col], 
            y=df[y_col1],
            marker=dict(color='#1f77b4'),  # Color for the first bar
            offsetgroup=0
        ),
        go.Bar(
            name=y_col2, 
            x=df[x_col], 
            y=df[y_col2],
            marker=dict(color='#ff7f0e'),  # Color for the second bar
            yaxis='y2',  # Secondary axis for the second variable
            offsetgroup=1
        )
    ])

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title=x_col,
        yaxis_title=y_col1,
        yaxis2=dict(
            title=y_col2,
            overlaying='y',
            side='right',
        ),
        template="plotly_white",
        barmode='group', 
        bargap=0.15,  # Gap between bars
        xaxis=dict(tickangle=45),  # Rotate x-axis labels
    )
    
    fig.show()




# def evaluate_model(model, X_train, y_train, X_test, y_test, products_to_keep, labels_dict):
#     # Compute predictions on the test set
#     y_test_pred = model.predict(X_test)
    
#     # Compute overall regression metrics for the entire test set
#     r2 = r2_score(y_test, y_test_pred)
#     mae = mean_absolute_error(y_test, y_test_pred)
#     mse = mean_squared_error(y_test, y_test_pred)
#     rmse = np.sqrt(mse)

#     # Print the overall metrics
#     print('Test set R^2:', r2)
#     print('Test set MAE:', mae)
#     print('Test set MSE:', mse)
#     print('Test set RMSE:', rmse)

#     print()
#     print()

#     # Collect errors for visualization
#     errors = {
#         'MAE': np.abs(y_test - y_test_pred),  # The absolute errors for MAE
#         'MSE': (y_test - y_test_pred)**2,  # Squared errors for MSE
#         'RMSE': np.sqrt((y_test - y_test_pred)**2)  # RMSE is the square root of squared errors
#     }

#     # Create a DataFrame to hold the errors for each metric
#     errors_df = pd.DataFrame(errors)

#     # Print the .describe() summary statistics for each metric
#     print("Summary statistics for errors (all samples):")
#     print(errors_df.describe())

#     print()
#     print()

#     # Plot the distribution of the errors for each metric using Plotly Express
#     fig = px.box(errors_df, labels={'variable': 'Metric', 'value': 'Error'}, title="Distribution of Errors for Each Metric")
#     fig.show()

#     # Initialize a list to store product-specific performance metrics
#     product_metrics = []

#     # Loop through each product in `products_to_keep` and evaluate the performance
#     for product in products_to_keep:    
#         X_test_product = X_test[X_test['ProductCode_encoded'] == product]
#         y_test_product = y_test[X_test_product.index]
        
#         product_name = labels_dict['ProductCode'][product]
        
#         # Get the model score (R²) for the product-specific data
#         y_test_product_pred = model.predict(X_test_product)
        
#         # Calculate R², MAE, MSE, and RMSE for this product
#         r2_product = r2_score(y_test_product, y_test_product_pred)
#         mae_product = mean_absolute_error(y_test_product, y_test_product_pred)
#         mse_product = mean_squared_error(y_test_product, y_test_product_pred)
#         rmse_product = np.sqrt(mse_product)
        
#         # Append the results to the list
#         product_metrics.append([product_name, r2_product, mae_product, mse_product, rmse_product])

#     # Create a DataFrame to hold the product-specific performance metrics
#     product_metrics_df = pd.DataFrame(product_metrics, columns=['Product Name', 'R^2', 'MAE', 'MSE', 'RMSE'])

#     # Print the DataFrame to show the metrics for each product
#     print("\nPerformance metrics for each product:")
#     print(product_metrics_df)

#     # Optionally, you can filter the low-performance products (e.g., where R^2 < 0.8)
#     low_performance_df = product_metrics_df[product_metrics_df['R^2'] < 0.8]

#     # Print the low-performance products
#     print("\nLow performance products (R^2 < 0.8):")
#     print(low_performance_df)

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_train, y_train, X_test, y_test, products_to_keep, labels_dict):
    # Compute predictions on the test set (in log scale)
    y_test_pred_log = model.predict(X_test)
    
    # Exponentiate the predictions to return to the original scale
    y_test_pred = np.exp(y_test_pred_log)
    
    # Exponentiate the true values to return to the original scale (if they were log-transformed)
    y_test_original = np.exp(y_test)

    # Compute overall regression metrics for the entire test set
    r2 = r2_score(y_test_original, y_test_pred)
    mae = mean_absolute_error(y_test_original, y_test_pred)
    mse = mean_squared_error(y_test_original, y_test_pred)
    rmse = np.sqrt(mse)
    
    # Compute MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_original - y_test_pred) / y_test_original)) * 100  # In percentage

    # Print the overall metrics
    print('Test set R^2:', r2)
    print('Test set MSE:', mse)
    print('Test set MAPE:', mape)

    print()
    print()

    # Collect errors for visualization (on original scale)
    errors = {

        'RMSE': np.sqrt((y_test_original - y_test_pred)**2),  # RMSE is the square root of squared errors
        'MAPE': np.abs((y_test_original - y_test_pred) / y_test_original) * 100  # MAPE in percentage
    }

    # Create a DataFrame to hold the errors for each metric
    errors_df = pd.DataFrame(errors)

    # Print the .describe() summary statistics for each metric
    print("Summary statistics for errors (all samples):")
    print(errors_df.describe())

    print()
    print()

    # Plot the distribution of the errors for each metric using Plotly Express
    fig = px.box(errors_df, labels={'variable': 'Metric', 'value': 'Error'}, title="Distribution of Errors for Each Metric")
    fig.show()

    # Initialize a list to store product-specific performance metrics
    product_metrics = []

    # Loop through each product in `products_to_keep` and evaluate the performance
    for product in products_to_keep:    
        X_test_product = X_test[X_test['ProductCode_encoded'] == product]
        y_test_product = y_test[X_test_product.index]
        
        product_name = labels_dict['ProductCode'][product]
        
        # Get the model score (R²) for the product-specific data
        y_test_product_pred_log = model.predict(X_test_product)
        
        # Exponentiate the predictions for this product to the original scale
        y_test_product_pred = np.exp(y_test_product_pred_log)
        
        # Exponentiate the true values for this product to the original scale
        y_test_product_original = np.exp(y_test_product)
        
        # Calculate R², MAE, MSE, RMSE, and MAPE for this product
        r2_product = r2_score(y_test_product_original, y_test_product_pred)
        mae_product = mean_absolute_error(y_test_product_original, y_test_product_pred)
        mse_product = mean_squared_error(y_test_product_original, y_test_product_pred)
        rmse_product = np.sqrt(mse_product)
        mape_product = np.mean(np.abs((y_test_product_original - y_test_product_pred) / y_test_product_original)) * 100  # MAPE in percentage
        
        # Append the results to the list
        product_metrics.append([product_name, r2_product, rmse_product, mape_product])

    # Create a DataFrame to hold the product-specific performance metrics
    product_metrics_df = pd.DataFrame(product_metrics, columns=['Product Name', 'R^2', 'RMSE', 'MAPE'])

    # Print the DataFrame to show the metrics for each product
    print("\nPerformance metrics for each product:")
    print(product_metrics_df)

    # Optionally, you can filter the low-performance products (e.g., where R^2 < 0.8)
    low_performance_df = product_metrics_df[product_metrics_df['R^2'] < 0.8]

    # Print the low-performance products
    print("\nLow performance products (R^2 < 0.8):")
    print(low_performance_df)
