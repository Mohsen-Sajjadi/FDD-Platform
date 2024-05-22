import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objs as go
import io
import base64
from pandas.errors import ParserError, EmptyDataError
from dash.exceptions import PreventUpdate
import operator
from datetime import datetime
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Initialize Dash app with Bootstrap external_stylesheets
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Include external CSS file for custom styles
app.css.append_css({"external_url": "/mnt/data/styles.css"})

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

# Define layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.Img(src=app.get_asset_url('Triton.png'), className='logo', style={'maxWidth': '160px', 'height': 'auto', 'margin-right': '10px'}), width='auto'),
        dbc.Col(html.Div([
            html.H1("Triton Concepts Analytics Platform", style={'margin-bottom': '4px'}),
            html.P("Fault Detection & Diagnostics", className='fdd-text', style={'font-size': '1.2em', 'font-weight': 'bold', 'margin-bottom': '10px'}),
        ], style={'background-color': '#343A40', 'color': '#FFFFFF', 'padding': '20px', 'border-radius': '10px', 'text-align': 'center', 'width': '100%'}), width=True),
    ], className='mb-4', style={'display': 'flex', 'align-items': 'center'}),

    dbc.Row([
        dbc.Col(html.Div([
            dcc.Upload(
                id='upload-data',
                children=[
                    html.Button('Import File', id='upload-button', className='btn btn-primary'),
                    html.Div(id='file-name-output', className='file-name-output')
                ],
                multiple=False
            )
        ]), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.Label("Select Date Range:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                display_format='YYYY-MM-DD HH:mm:ss',
                className='form-control modern-datepicker'
            )
        ]), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.Label("Control Points:", style={'color': '#FFFFFF', 'font-family': 'Arial', 'font-size': '20px'}),
            dcc.Dropdown(
                id='control-points-dropdown',
                options=[],
                multi=True,
                placeholder="Search and select control points...",
                className='form-control',
                style={'width': '100%'}
            )
        ]), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.Button('Select All Control Points', id='select-all-button', className='btn btn-primary mr-2'),
            html.Button('Control Points', id='open-modal-button', className='btn btn-secondary mr-2'),
            html.Button('Reset', id='reset-button', className='btn btn-danger')
        ]), width=12, className='mb-4 text-left buttons-container'),
    ]),

    dbc.Modal([
        dbc.ModalHeader("Select Control Points"),
        dbc.ModalBody([
            dcc.Dropdown(
                id='modal-control-points-search',
                options=[],
                placeholder="Search control points...",
                multi=True,
                style={'margin-bottom': '20px', 'background-color': 'black', 'color': 'white'}
            ),
            dcc.Checklist(
                id='control-points-checklist',
                options=[],
                style={'maxHeight': '200px', 'overflowY': 'scroll', 'width': '100%'}
            )
        ]),
        dbc.ModalFooter(
            html.Button("Close", id="close-modal-button", className="btn btn-secondary")
        )
    ], id="modal", is_open=False),

    dbc.Row([
        dbc.Col(dcc.Graph(id='control-points-graph-1plus', config={'scrollZoom': False}, className='control-points-graph'), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='control-points-graph-01', config={'scrollZoom': False}, className='control-points-graph'), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.Label('Desired Value:'),
            dcc.Input(id='desired-value', type='text', value=None, className='form-control'),
        ]), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='selected-points-output', className='selected-points-output'), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='time-under-condition', className='time-under-condition'), width=12, className='mb-4'),
    ]),

    dbc.Row([
        dbc.Col(html.Div([
            html.Label('Condition:'),
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
                className='form-control'
            )
        ]), width=12, className='mb-4'),
    ]),
])

# Callback to upload file and update dataframe
@app.callback(
    [Output('control-points-dropdown', 'options'),
     Output('control-points-checklist', 'options'),
     Output('modal-control-points-search', 'options'),
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
        return [], [], [], None, None, f"Error: Unable to parse file '{filename}': {str(e)}"

    # Convert timestamp column to datetime
    # Assuming the column name is 'Time stamp' or similar
    timestamp_column = next((col for col in df.columns if 'time' in col.lower()), None)
    if timestamp_column is None:
        return [], [], [], None, None, f"Error: Timestamp column not found in the file."
    
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    # Set Timestamp as index
    df.set_index(timestamp_column, inplace=True)
    # Sort DataFrame by index
    df.sort_index(inplace=True)
    
    control_points = [{'label': col, 'value': col} for col in df.columns]

    start_date = df.index.min()
    end_date = df.index.max()

    # Change the label of the button to include the filename
    upload_button_label = f'Import File: {filename}' if filename else 'Import File'

    return control_points, control_points, control_points, start_date, end_date, upload_button_label

# Callback to update graph based on selected control points and time frame
@app.callback(
    [Output('control-points-graph-01', 'figure'),
     Output('control-points-graph-1plus', 'figure')],
    [Input('control-points-dropdown', 'value'),
     Input('control-points-checklist', 'value'),
     Input('modal-control-points-search', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('desired-value', 'value'),
     Input('condition-dropdown', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_control_points_graph(selected_dropdown_points, selected_checklist_points, selected_modal_points, start_date, end_date, desired_value, condition, reset_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Determine if the reset button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'reset-button':
        return {'data': [], 'layout': go.Layout(title='Trends of Control Points (0-1)', xaxis=dict(title='Timestamp'), yaxis=dict(title='Value'))}, \
               {'data': [], 'layout': go.Layout(title='Trends of Control Points (1+)', xaxis=dict(title='Timestamp'), yaxis=dict(title='Value'))}

    selected_control_points = list(set(selected_dropdown_points or []) | set(selected_checklist_points or []) | set(selected_modal_points or []))
    if not selected_control_points or start_date is None or end_date is None:
        return {'data': [], 'layout': go.Layout(title='Trends of Control Points (0-1)', xaxis=dict(title='Timestamp'), yaxis=dict(title='Value'))}, \
               {'data': [], 'layout': go.Layout(title='Trends of Control Points (1+)', xaxis=dict(title='Timestamp'), yaxis=dict(title='Value'))}

    filtered_df = df.loc[start_date:end_date]
    traces_01 = []  # For values between 0 and 1
    traces_1plus = []  # For values greater than 1

    shapes = []  # Initialize shapes list

    # Convert selected columns to numeric type
    for column in selected_control_points:
        filtered_df[column] = pd.to_numeric(filtered_df[column], errors='coerce')

    # Add traces for selected control points
    for column in selected_control_points:
        if filtered_df[column].between(0, 1).all():
            trace = go.Scatter(
                x=filtered_df.index,
                y=filtered_df[column],
                mode='lines',
                name=f"{column} (0-1)",
                hoverinfo='text',
                hovertemplate='%{y:.2f}'
            )
            traces_01.append(trace)
        else:
            trace = go.Scatter(
                x=filtered_df.index,
                y=filtered_df[column],
                mode='lines',
                name=column,
                hoverinfo='text',
                hovertemplate='%{y:.2f}'
            )
            traces_1plus.append(trace)

        # Calculate shaded regions only if desired_value and condition are not None or empty
        if desired_value and condition:
            try:
                desired_value_float = float(desired_value)
            except ValueError:
                # Handle invalid desired value (non-numeric input)
                return {'data': [], 'layout': {}}, {'data': [], 'layout': {}}

            y_values = filtered_df[column]
            indices = OPERATORS.get(condition, operator.gt)(y_values, desired_value_float)
            time_difference = filtered_df.index[1] - filtered_df.index[0]

            # Initialize variables to track shaded region
            start_index = None
            end_index = None
            for i, ind in enumerate(indices):
                if ind and start_index is None:
                    start_index = i
                elif not ind and start_index is not None:
                    end_index = i
                    # Add shaded region shape
                    shapes.append({
                        'type': 'rect',
                        'xref': 'x',
                        'yref': 'paper',
                        'x0': filtered_df.index[start_index],
                        'y0': 0,
                        'x1': filtered_df.index[end_index],
                        'y1': 1,
                        'fillcolor': 'rgba(255, 0, 0, 0.3)',  # Adjust color and opacity as needed
                        'line': {'width': 0},
                        'layer': 'below'
                    })
                    start_index = None
                    end_index = None

    layout = go.Layout(
        title='Trends of Control Points (0-1)',
        xaxis=dict(title='Timestamp', tickfont=dict(color='#FFFFFF')),
        yaxis=dict(title='Value', tickfont=dict(color='#FFFFFF')),
        showlegend=True,
        margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        hovermode='x unified',
        plot_bgcolor='#202020',
        paper_bgcolor='#1E1E1E',
        legend=dict(font=dict(color='#CCCCCC')),
        shapes=shapes  # Add shaded regions to the layout
    )

    layout_1plus = go.Layout(
        title='Trends of Control Points (1+)',
        xaxis=dict(title='Timestamp', tickfont=dict(color='#FFFFFF')),
        yaxis=dict(title='Value', tickfont=dict(color='#FFFFFF')),
        showlegend=True,
        margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        hovermode='x unified',
        plot_bgcolor='#202020',
        paper_bgcolor='#1E1E1E',
        legend=dict(font=dict(color='#FFFFFF')),
        shapes=shapes  # Add shaded regions to the layout
    )

    return {'data': traces_01, 'layout': layout}, {'data': traces_1plus, 'layout': layout_1plus}

# Callback to display selected points and total time under condition
@app.callback(
    [Output('selected-points-output', 'children'),
     Output('time-under-condition', 'children')],
    [Input('control-points-dropdown', 'value'),
     Input('control-points-checklist', 'value'),
     Input('modal-control-points-search', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('desired-value', 'value'),
     Input('condition-dropdown', 'value')]
)
def display_selected_points(selected_dropdown_points, selected_checklist_points, selected_modal_points, start_date, end_date, desired_value, condition):
    if not (selected_dropdown_points or selected_checklist_points or selected_modal_points) or start_date is None or end_date is None or condition is None or desired_value is None or desired_value == '':
        return '', ''

    selected_control_points = list(set(selected_dropdown_points or []) | set(selected_checklist_points or []) | set(selected_modal_points or []))
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

# Callback to select all control points and reset control points
@app.callback(
    [Output('control-points-dropdown', 'value'),
     Output('control-points-checklist', 'value'),
     Output('modal-control-points-search', 'value')],
    [Input('select-all-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('control-points-dropdown', 'options'),
     State('control-points-checklist', 'options'),
     State('modal-control-points-search', 'options')]
)
def manage_control_points(select_all_clicks, reset_clicks, dropdown_options, checklist_options, modal_options):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Determine which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'select-all-button':
        dropdown_values = [option['value'] for option in dropdown_options]
        checklist_values = [option['value'] for option in checklist_options]
        modal_values = [option['value'] for option in modal_options]
        return dropdown_values, checklist_values, modal_values
    elif button_id == 'reset-button':
        return [], [], []

# Callback to toggle modal
@app.callback(
    Output('modal', 'is_open'),
    [Input('open-modal-button', 'n_clicks'), Input('close-modal-button', 'n_clicks')],
    [State('modal', 'is_open')]
)
def toggle_modal(open_click, close_click, is_open):
    if open_click or close_click:
        return not is_open
    return is_open

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8051)
