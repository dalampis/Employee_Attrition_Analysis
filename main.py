import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

file_path = 'employees_attrition_updated.csv'
data = pd.read_csv(file_path)
print('The data that we want to analyze:')
print(data)
print()

numeric_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
                   'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

corr_matrix = np.corrcoef(data[numeric_columns].dropna(), rowvar = False)
print('The correlation matrix of the numerical columns:')
print(corr_matrix)
print()

plt.figure(figsize = (12, 8))
sns.heatmap(corr_matrix, annot = True, xticklabels = numeric_columns, yticklabels = numeric_columns, cmap = 'coolwarm')
plt.title('The Correlation Matrix Heatmap of Employees Attrition')
plt.show()

filtered_data = data[data['JobRole'].isin(['Sales Executive', 'Research Scientist', 'Laboratory Technician']) &
    data['Department'].isin(['Research & Development', 'Sales'])]
grouped_data = filtered_data.groupby(['YearsAtCompany', 'JobRole', 'Department']).agg({
    'MonthlyIncome': ['mean', 'std', 'count']
}).reset_index()
grouped_data.columns = ['YearsAtCompany', 'JobRole', 'Department', 'meanMonthlyIncome', 'stdMonthlyIncome', 'count']
grouped_data = grouped_data[grouped_data['count'] >= 5]
print('The data that we want to visualize:')
print(grouped_data)
print()

plt.figure(figsize = (12, 8))
sns.lineplot(data = grouped_data, x = 'YearsAtCompany', y = 'meanMonthlyIncome', hue = 'Department', style = 'JobRole',
             marker = True)
for job_role in grouped_data['JobRole'].unique():
    role_data = grouped_data[grouped_data['JobRole'] == job_role]
    plt.errorbar(role_data['YearsAtCompany'], role_data['meanMonthlyIncome'], yerr = role_data['stdMonthlyIncome'], fmt = 'x', label = job_role, capsize = 6)
plt.title('The trend of average monthly income over years at the company')
plt.xlabel('Years at Company')
plt.ylabel('Average Monthly Income')
plt.legend(title = 'Department and Job Role', loc = 'upper right')
plt.grid(True)
plt.show()

new_df = data[['Age', 'TotalWorkingYears', 'MonthlyIncome']].dropna()

# Perform multiple linear regression using scipy
X = new_df[['Age', 'TotalWorkingYears']]
y = new_df['MonthlyIncome']

# Add a constant column for the intercept
X = np.column_stack((np.ones(X.shape[0]), X))
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

# Define the regression plane
def regression_plane(x, y, beta):
    return beta[0] + beta[1] * x + beta[2] * y

# Generate a grid of values
x_range = np.linspace(X[:, 1].min(), X[:, 1].max(), num=50)
y_range = np.linspace(X[:, 2].min(), X[:, 2].max(), num=50)
x_grid, y_grid = np.meshgrid(x_range, y_range)
z_grid = regression_plane(x_grid, y_grid, beta)

# Create a 3D scatter plot using plotly
scatter = go.Scatter3d(
    x=new_df['Age'],
    y=new_df['TotalWorkingYears'],
    z=new_df['MonthlyIncome'],
    mode='markers',
    marker=dict(
        size=5,
        color=new_df['MonthlyIncome'],
        colorscale='Bluered',
        opacity=0.8
    )
)

# Create the regression plane
plane = go.Surface(
    x=x_grid,
    y=y_grid,
    z=z_grid,
    colorscale='Blues',
    opacity=0.5
)

# Define the layout of the plot
layout = go.Layout(
    title='3D Scatter Plot of Monthly Income with Regression Plane',
    scene=dict(
        xaxis_title='Age',
        yaxis_title='Total Working Years',
        zaxis_title='Monthly Income'
    )
)

# Combine the scatter plot and plane into a figure
fig = go.Figure(data=[scatter, plane], layout=layout)

# Show the plot
fig.show()