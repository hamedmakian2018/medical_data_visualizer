import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

  # Import data
df = pd.read_csv('medical_examination.csv')

  # Add 'overweight' column
df['overweight'] = np.where(df['weight'] / (df['height'] / 100) ** 2 >= 25, 1, 0)

  # Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)


  # Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  vars = sorted(\
    ["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])
  df_cat = df.melt(id_vars=['cardio'],\
    value_vars=vars)


  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = pd.DataFrame(df_cat.groupby(["variable", "value","cardio"])\
        ["value"].count()).rename(columns={"value": "total"})\
        .reset_index()



  # Draw the catplot with 'sns.catplot()'
  fig = plt.subplots(figsize=(10, 6))
  fig=sns.catplot(data=df_cat, x="variable",
                 y="total",
                 col="cardio",
                 hue="value",
                 kind="bar",
                 order=vars
                 ).fig



  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig




  # Draw Heat Map
def draw_heat_map():
      # Clean the data
      df_heat = df[
      (df['ap_lo'] <= df['ap_hi'])
      & (df['height'] >= 
      df['height'].quantile(0.025))
      & (df['height'] <= 
      df['height'].quantile(0.975))
      & (df['weight'] >= 
      df['weight'].quantile(0.025))
      & (df['weight'] <= 
     df['weight'].quantile(0.975))
      ]
      #names = df_heat.columns.tolist()
      #names[names.index('sex')] = 'gender'
      #df_heat.columns = names
      # Calculate the correlation matrix
      corr = df_heat.corr()

      # Generate a mask for the upper triangle
      #mask = np.triu(np.ones_like(corr, dtype=bool))
      #mask[np.diag_indices_from(mask)] = False
      mask = np.triu(corr)
      


      # Set up the matplotlib figure
      fig, ax= plt.subplots(figsize=(11, 9))

      # Draw the heatmap with 'sns.heatmap()'
   
      sns.heatmap(corr, mask=mask,annot=True, 
      fmt='.1f',center=0, 
      vmin=-0.15, vmax=0.3,  square=True, linewidths = 0.7 ,
      cbar_kws={'shrink':0.5,'ticks': [-0.08,0.00,0.08,0.16,0.24], 
      })
      


      # Do not modify the next two lines
      fig.savefig('heatmap.png')
      return fig
