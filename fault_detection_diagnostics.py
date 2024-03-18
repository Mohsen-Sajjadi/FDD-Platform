"""
Created on Thu Mar 14 15:58:10 2024

@author: MohsenSajjadi
"""
import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import io
import base64
from pandas.errors import ParserError, EmptyDataError
from dash.exceptions import PreventUpdate
import operator
from datetime import datetime

# Initialize Dash app
app = dash.Dash(__name__)

# Initialize DataFrame
df = pd.DataFrame()

OPERATORS = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '>': operator.gt
}

# Define layout with dark theme
app.layout = html.Div(style={'backgroundColor': '#1E1E1E', 'color': '#FFFFFF', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.Div([
        html.H1("Triton Concepts Analytics Platform", style={'textAlign': 'left', 'marginTop': '10px', 'marginLeft': '20px', 'fontSize': '32px'}),
        html.Img(src=app.get_asset_url('Triton.png'), style={'height':'50px', 'float':'right', 'marginTop': '10px', 'marginRight': '20px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'backgroundColor': '#343A40', 'padding': '10px'}),

    html.Div([
        html.P("Fault Detection & Diagnostics", style={'textAlign': 'left', 'marginTop': '-10px', 'marginLeft': '20px', 'fontSize': '24px'}),
    ], style={'marginTop': '20px', 'marginLeft': '20px'}),

    html.Div([
        dcc.Upload(
            id='upload-data',
            children=[
                html.Button('Import File', id='upload-button', style={'backgroundColor': '#007BFF', 'color': '#FFFFFF', 'fontSize': '16px', 'padding': '12px 24px'}),
                html.Div(id='file-name-output', style={'marginLeft': '20px', 'marginTop': '5px'})
            ],
            multiple=False
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px', 'marginTop': '20px'}),

    # Date picker range with time
    html.Div([
        html.Label("Select Date Range:", style={'color': '#FFFFFF', 'fontWeight': 'bold'}),
        dcc.DatePickerRange(
            id='date-picker-range',
            display_format='YYYY-MM-DD HH:mm:ss',
            style={'color': '#000080', 'width': '100%'}  # Adjusted width for longer tap
        ),
    ], style={'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '20px'}),

    # Dropdown for selecting all control points
    html.Div([
        html.Label("Control Points:", style={'color': '#FFFFFF', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='control-points-dropdown',
            multi=True,
            style={'color': '#000080', 'backgroundColor': 'black'}
        )
    ], style={'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '20px'}),

    # Graph for displaying trends of selected control points
    dcc.Graph(id='control-points-graph', config={'scrollZoom': False}, style={'backgroundColor': '#202020', 'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '20px'}),

    # Display selected points
    html.Div(id='selected-points-output', style={'marginTop': '20px', 'marginLeft': '20px'}),
    
    # Display total time under condition
    html.Div(id='time-under-condition', style={'marginTop': '20px', 'marginLeft': '20px'}),

    # Input for desired value
    html.Div([
        html.Label('Desired Value:', style={'color': '#FFFFFF', 'fontWeight': 'bold'}),
        dcc.Input(id='desired-value', type='text', value=None),
    ], style={'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '20px'}),

    # Dropdown for selecting condition
    html.Div([
        html.Label('Condition:', style={'color': '#FFFFFF', 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='condition-dropdown',
            options=[
                {'label': 'Less Than', 'value': '<'},
                {'label': 'Less Than or Equal To', 'value': '<='},
                {'label': 'Equal To', 'value': '=='},
                {'label': 'Not Equal To', 'value': '!='},
                {'label': 'Greater Than or Equal To', 'value': '>='},
                {'label': 'Greater Than', 'value': '>'}
            ],
            value='>',
            style={'color': '#000080', 'backgroundColor': 'black'}
        )
    ], style={'marginLeft': '20px', 'marginRight': '20px', 'marginTop': '20px'}),
])

# Callback to upload file and update dataframe
@app.callback(
    [Output('control-points-dropdown', 'options'),
     Output('date-picker-range', 'start_date'),
     Output('date-picker-range', 'end_date'),
     Output('file-name-output', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
def update_output(contents, filename):
    global df  # Access the global DataFrame
    if contents is None:
        raise PreventUpdate
    
    content_type, content_string = contents.split(',')
    decoded = io.StringIO(base64.b64decode(content_string).decode('utf-8'))
    try:
        # Read CSV with NaN values
        df = pd.read_csv(decoded, skipinitialspace=True)
    except (ParserError, EmptyDataError) as e:
        return [], None, None, f"Error: Unable to parse file '{filename}': {str(e)}"

    # Convert timestamp column to datetime
    # Assuming the column name is 'Time stamp' or similar
    timestamp_column = next((col for col in df.columns if 'time' in col.lower()), None)
    if timestamp_column is None:
        return [], None, None, f"Error: Timestamp column not found in the file."
    
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    # Set Timestamp as index
    df.set_index(timestamp_column, inplace=True)
    # Sort DataFrame by index
    df.sort_index(inplace=True)
    
    control_points = []
    
    # Iterate over columns to categorize them
    for col in df.columns:
        control_points.append({'label': col, 'value': col})

    start_date = df.index.min()
    end_date = df.index.max()

    # Change the label of the button to include the filename
    upload_button_label = f'Import File: {filename}' if filename else 'Import File'

    return control_points, start_date, end_date, upload_button_label

# Callback to update graph based on selected control points and time frame
@app.callback(
    Output('control-points-graph', 'figure'),
    [Input('control-points-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('desired-value', 'value'),
     Input('condition-dropdown', 'value')]
)
def update_control_points_graph(selected_control_points, start_date, end_date, desired_value, condition):
    if not selected_control_points or start_date is None or end_date is None:
        return {'data': [], 'layout': {}}

    filtered_df = df.loc[start_date:end_date]
    traces = []
    shapes = []
    # Add traces for selected control points
    for column in selected_control_points:
        trace = go.Scatter(
            x=filtered_df.index,
            y=filtered_df[column],
            mode='lines',
            name=column,
            hoverinfo='text',
            hovertemplate='%{y:.2f}'
        )
        traces.append(trace)
        
        if desired_value is not None and condition is not None and desired_value != '':
            y_values = filtered_df[column]
            indices = OPERATORS[condition](y_values, float(desired_value))
            in_condition = False
            for i, ind in enumerate(indices):
                if ind and not in_condition:
                    start = filtered_df.index[i]
                    in_condition = True
                    print(f"Start Time: {start}")
                elif not ind and in_condition:
                    end = filtered_df.index[i]
                    in_condition = False
                    print(f"End Time: {end}")
                    shapes.append(dict(type="rect", xref="x", yref="paper", x0=start, y0=0, x1=end, y1=1, fillcolor="red", opacity=0.5, layer="below", line_width=0))

    layout = go.Layout(
        title='Trends of Control Points',
        xaxis=dict(title='Timestamp', tickfont=dict(color='#FFFFFF')),
        yaxis=dict(title='Value', tickfont=dict(color='#FFFFFF')),
        showlegend=True,
        margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        hovermode='x unified',
        plot_bgcolor='#202020',
        paper_bgcolor='#1E1E1E',
        legend=dict(font=dict(color='#CCCCCC')),
        shapes=shapes
    )

    return {'data': traces, 'layout': layout}

# Define callback to display selected points and total time under condition
@app.callback(
    [Output('selected-points-output', 'children'),
     Output('time-under-condition', 'children')],
    [Input('control-points-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('desired-value', 'value'),
     Input('condition-dropdown', 'value')]
)
def display_selected_points(selected_control_points, start_date, end_date, desired_value, condition):
    if not selected_control_points or start_date is None or end_date is None or condition is None or desired_value is None or desired_value == '':
        return '', ''

    filtered_df = df.loc[start_date:end_date]
    selected_points_info = []
    total_time_info = []

    for column in selected_control_points:
        if column in filtered_df.columns:
            # Calculate average with 2 decimal places
            average_value = round(filtered_df[column].mean(), 2)

            # Calculate total time under condition
            y_values = filtered_df[column]
            indices = OPERATORS[condition](y_values, float(desired_value))
            time_difference = filtered_df.index[1] - filtered_df.index[0]
            time_difference_in_minutes = time_difference.total_seconds() / 60

            # Initialize a counter for total time under condition
            total_time_under_condition = 0
            
            # Iterate over the indices to calculate total time under condition
            in_condition = False
            for i, ind in enumerate(indices):
                if ind and not in_condition:
                    start_time = filtered_df.index[i]
                    in_condition = True
                elif not ind and in_condition:
                    end_time = filtered_df.index[i]
                    in_condition = False
                    # Calculate time under condition for this interval and add it to the total
                    total_time_under_condition += (end_time - start_time).total_seconds() / 60
            total_time_info.append(html.Div(f"Total time under condition for {column}: {total_time_under_condition} minutes"))

            selected_points_info.append(html.Div([
                html.H4(f"Selected Control Point: {column}"),
                dash_table.DataTable(
                    id={
                        'type': 'datatable',
                        'index': column
                    },
                    columns=[
                        {'name': 'Metric', 'id': 'metric'},
                        {'name': 'Value', 'id': 'value'}
                    ],
                    data=[
                        {'metric': 'Min Value', 'value': filtered_df[column].min()},
                        {'metric': 'Max Value', 'value': filtered_df[column].max()},
                        {'metric': 'Average Value', 'value': average_value}
                    ],
                    style_cell={'textAlign': 'left', 'backgroundColor': '#343A40', 'color': '#FFFFFF'},
                    style_header={'backgroundColor': '#007BFF', 'fontWeight': 'bold'}
                ),
            ]))

    return selected_points_info, total_time_info

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8051)