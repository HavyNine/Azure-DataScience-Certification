# Azure Data Scientist

Prazo: February 16, 2023 → March 14, 2023
Status: In progress

# **Explorar e analisar dados com o Python**

# ****Visualizar os dados com o Matplotlib****

Docs:

- [Matplolib](https://matplotlib.org/stable/users/index.html)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Numpy](https://numpy.org/doc/stable/)

- Base

```python
import pandas as pd

# Load data from a text file
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/grades.csv
df_students = pd.read_csv('grades.csv',delimiter=',',header='infer')

# Remove any rows with missing data
df_students = df_students.dropna(axis=0, how='any')

# Calculate who passed, assuming '60' is the grade needed to pass
passes  = pd.Series(df_students['Grade'] >= 60)

# Save who passed to the Pandas dataframe
df_students = pd.concat([df_students, passes.rename("Pass")], axis=1)

# Print the result out into this notebook
df_students
```

- ****Visualizing data with Matplotlib****

```python
# Ensure plots are displayed inline in the notebook
%matplotlib inline

from matplotlib import pyplot as plt

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade)

# Display the plot
plt.show()
```

```python
# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Display the plot
plt.show()
```

- A plot is technically contained within a **Figure**. In the previous examples, the figure was created implicitly for you, but you can create it explicitly. For example, the following code creates a figure with a specific size.

```python
# Create a Figure
fig = plt.figure(figsize=(8,3))

# Create a bar plot of name vs grade
plt.bar(x=df_students.Name, height=df_students.Grade, color='orange')

# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Show the figure
plt.show()
```

- Subplots

```python
# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students.Name, height=df_students.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()
```

- Intrinsic methods in pandas

```python
df_students.plot.bar(x='Name', y='StudyHours', color='teal', figsize=(6,4))
```

****Getting started with statistical analysis****

- Descriptive statistics and data distribution

```python
# Get the variable to examine
var_data = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var_data)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
```

- ****Measures of central tendency****
    - The *mean*: A simple average based on adding together all of the values in the sample set and then dividing the total by the number of samples.
    - The *median*: The value in the middle of the range of all of the sample values.
    - The *mode*: The most commonly occurring value in the sample set*.
        - *Of course, in some sample sets, there may be a tie for the most common value. In those cases, the dataset is described as *bimodal* or even *multimodal*.

```python
# Get the variable to examine
var = df_students['Grade']

# Get statistics
min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]

print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                        mean_val,
                                                                                        med_val,
                                                                                        mod_val,
                                                                                        max_val))

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var)

# Add lines for the statistics
plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
```

- Another way to visualize the distribution of a variable is to use a *box* plot (sometimes called a *box-and-whiskers* plot).

```python
# Get the variable to examine
var = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.boxplot(var)

# Add titles and labels
plt.title('Data Distribution')

# Show the figure
fig.show()
```

- For learning, it can be useful to combine histograms and box plots, with the box plot's orientation changed to align it with the histogram. (In some ways, it can be helpful to think of the histogram as a "front elevation" view of the distribution, and the box plot as a "plan" view of the distribution from above.)

```python
# Create a function that we can re-use
def show_distribution(var_data):
    from matplotlib import pyplot as plt

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle('Data Distribution')

    # Show the figure
    fig.show()

# Get the variable to examine
col = df_students['Grade']
# Call the function
show_distribution(col)
```

- If we have enough samples, we can calculate something called a *probability density function*
, which estimates the distribution of grades for the full population. The **pyplot** class from Matplotlib provides a helpful plot function to show this density.

```python
def show_density(var_data):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title('Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of Grade
col = df_students['Grade']
show_density(col)
```

# ****Examinar os dados do mundo real****

Lembre-se de que os dados do mundo real sempre terão problemas, mas geralmente é algo que pode ser contornado. Lembre-se de:

- Verifique se há valores ausentes e dados registrados incorretamente
- Considere a remoção de outliers óbvios
- Considere quais fatores do mundo real podem afetar sua análise e se o tamanho dos conjuntos de dados é suficientemente grande para lidar com isso
- Verifique se os dados brutos estão tendenciosos e considere as opções de correção se encontrados.

****Exploring data with Python - real world data****

```python
# Get the variable to examine
col = df_students['StudyHours']
# Call the function
show_distribution(col)
```

- O seguinte código usa a função **quantile** do Pandas para excluir observações abaixo do percentil 0,01 (o valor acima do qual 99% dos dados residem).

```python
# calculate the 0.01th percentile
q01 = df_students.StudyHours.quantile(0.01)
# Get the variable to examine
col = df_students[df_students.StudyHours>q01]['StudyHours']
# Call the function
show_distribution(col)
```

- **Dica**: Você também pode eliminar valores discrepantes no extremo superior da distribuição definindo um limite em um valor percentual alto. Por exemplo, você pode usar a função **quantile** para encontrar o percentil 0,99 abaixo do qual 99% dos dados residem.

- Let's look at the density for this distribution.

```python
def show_density(var_data):
    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title('Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of StudyHours
show_density(col)
```

- Esse tipo de distribuição é chamado de *assimétrica à direita*. A massa dos dados está do lado esquerdo da distribuição, criando uma longa cauda para a direita devido aos valores no extremo superior, que puxam a média para a direita.

****Measures of variance****

Typical statistics that measure variability in the data include:

- **Range**: The difference between the maximum and minimum. There's no built-in function for this, but it's easy to calculate using the **min** and **max** functions.
- **Variance**: The average of the squared difference from the mean. You can use the built-in **var** function to find this.
- **Standard Deviation**: The square root of the variance. You can use the built-in **std** function to find this.

```python
for col_name in ['Grade','StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
```

- When working with a *normal* distribution, the standard deviation works with the particular characteristics of a normal distribution to provide even greater insight.

```python
import scipy.stats as stats

# Get the Grade column
col = df_students['Grade']

# get the density
density = stats.gaussian_kde(col)

# Plot the density
col.plot.density()

# Get the mean and standard deviation
s = col.std()
m = col.mean()

# Annotate 1 stdev
x1 = [m-s, m+s]
y1 = density(x1)
plt.plot(x1,y1, color='magenta')
plt.annotate('1 std (68.26%)', (x1[1],y1[1]))

# Annotate 2 stdevs
x2 = [m-(s*2), m+(s*2)]
y2 = density(x2)
plt.plot(x2,y2, color='green')
plt.annotate('2 std (95.45%)', (x2[1],y2[1]))

# Annotate 3 stdevs
x3 = [m-(s*3), m+(s*3)]
y3 = density(x3)
plt.plot(x3,y3, color='orange')
plt.annotate('3 std (99.73%)', (x3[1],y3[1]))

# Show the location of the mean
plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)

plt.axis('off')

plt.show()
```

- The horizontal lines show the percentage of data within 1, 2, and 3 standard deviations of the mean (plus or minus).
- In any normal distribution:
    - Approximately 68.26% of values fall within one standard deviation from the mean.
    - Approximately 95.45% of values fall within two standard deviations from the mean.
    - Approximately 99.73% of values fall within three standard deviations from the mean.
    

### ****Comparing data****

****Comparing numeric and categorical variables****

```python
df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))
```

****Comparing numeric variables****

```python
# Create a bar plot of name vs grade and study hours
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```

- A common technique when dealing with numeric data in different scales is to *normalize*
 the data so that the values retain their proportional distribution but are measured on the same scale. To accomplish this, we'll use a technique called *MinMax*
 scaling that distributes the values proportionally on a scale of 0 to 1.

```python
from sklearn.preprocessing import MinMaxScaler

# Get a scaler object
scaler = MinMaxScaler()

# Create a new dataframe for the scaled values
df_normalized = df_sample[['Name', 'Grade', 'StudyHours']].copy()

# Normalize the numeric columns
df_normalized[['Grade','StudyHours']] = scaler.fit_transform(df_normalized[['Grade','StudyHours']])

# Plot the normalized values
df_normalized.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```

> **Note**: Data scientists often quote the maxim "*correlation* is not *causation*". In other words, as tempting as it might be, you shouldn't interpret the statistical correlation as explaining *why* one of the values is high. In the case of the student data, the statistics demonstrate that students with high grades tend to also have high amounts of study time, but this is not the same as proving that they achieved high grades *because* they studied a lot. The statistic could equally be used as evidence to support the nonsensical conclusion that the students studied a lot *because* their grades were going to be high.
> 

```python
# Create a scatter plot
df_sample.plot.scatter(title='Study Time vs Grade', x='StudyHours', y='Grade')
```

- adding a *regression* line (or a *line of best fit*) to the plot that shows the general trend in the data. To do this, we'll use a statistical technique called *least squares regression*.

```python
from scipy import stats

#
df_regression = df_sample[['Grade', 'StudyHours']].copy()

# Get the regression slope and intercept
m, b, r, p, se = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
print('slope: {:.4f}\ny-intercept: {:.4f}'.format(m,b))
print('so...\n f(x) = {:.4f}x + {:.4f}'.format(m,b))

# Use the function (mx + b) to calculate f(x) for each x (StudyHours) value
df_regression['fx'] = (m * df_regression['StudyHours']) + b

# Calculate the error between f(x) and the actual y (Grade) value
df_regression['error'] = df_regression['fx'] - df_regression['Grade']

# Create a scatter plot of Grade vs StudyHours
df_regression.plot.scatter(x='StudyHours', y='Grade')

# Plot the regression line
plt.plot(df_regression['StudyHours'],df_regression['fx'], color='cyan')

# Display the plot
plt.show()
```

****Using the regression coefficients for prediction****

```python
# Define a function based on our regression coefficients
def f(x):
    m = 6.3134
    b = -17.9164
    return m*x + b

study_time = 14

# Get f(x) for study time
prediction = f(study_time)

# Grade can't be less than 0 or more than 100
expected_grade = max(0,min(100,prediction))

#Print the estimated grade
print ('Studying for {} hours per week may result in a grade of {:.0f}'.format(study_time, expected_grade))
```

# ****Treinar e avaliar modelos de regressão****

# ****Introdução****

- *Regressão* é quando os modelos preveem um número.
- Em machine learning, a meta da regressão é criar um modelo que possa prever um valor numérico e quantificável, como preço, valor, tamanho ou outro número escalar.

> O aprendizado de máquina é baseado em estatísticas e matemática, e é importante ter atenção a termos específicos que estatísticos e matemáticos (e, portanto, cientistas de dados) usam. Você pode considerar a diferença entre um valor de rótulo *previsto* e o valor *real* do rótulo como uma medida de ***erro***. No entanto, na prática, os valores "reais" são baseados em observações de exemplos (que podem estar sujeitos a uma variação aleatória). Essas observações de exemplos são normalmente avaliadas a partir de um conjunto de dados de validação, que são usados para testar a capacidade de um modelo de previsão de aprendizado de máquina para fazer previsões precisas. Para deixar claro que estamos comparando um *valor previsto* (*ŷ*) com um valor *observado* (*y*), nós nos referimos à diferença entre eles como os ***resíduos***. Estes resíduos são calculados a partir da diferença entre o valor previsto e o valor observado para cada observação de exemplo. Podemos resumir os resíduos de todas as previsões de dados de validação para calcular a ***perda*** geral no modelo como uma medida do desempenho de previsão. Essa medida de perda é importante porque ajuda a avaliar a eficácia de um modelo de aprendizado de máquina para prever valores observados.
>