import pandas as pd 
import numpy as np

tripvisorDf= pd.read_sql('call fetchTripVisor()', engine)

var_Columns = ['Review', 'Reviewer_Score']

tripvisorDfImp = pd.DataFrame(tripvisorDf[var_Columns])
tripvisorDfImp.info()

tripvisorDfImp.loc[tripvisorDfImp['Reviewer_Score'] == 6] = np.nan

tripvisorDfImp.dropna(subset=['Reviewer_Score'])

tripvisorDfImp.loc[tripvisorDfImp['Reviewer_Score'] < 6, 'Label'] = 'Negative' 
tripvisorDfImp.loc[tripvisorDfImp['Reviewer_Score'] > 6, 'Label'] = 'Positive' 

tripvisorDfImp.sample(5)

tripvisorDfImp.drop(columns=['Reviewer_Score'], inplace=True)

tripvisorDfImp.to_csv('scrapedCleaned.csv')

tripvisorDfImp.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

tripvisorDfImp['Total_Words'] = tripvisorDfImp['Review'].str.split().str.len()
tripvisorDfImp.drop(columns=['totalwords'], inplace=True)

total = pd.concat([tripvisorDfImp, badReviews, goodReviews])

total.reset_index(inplace=True, drop=True)

tripvisorDfImp.head(110)

total.to_csv('totalCleaned.csv')