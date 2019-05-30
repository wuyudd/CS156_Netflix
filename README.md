# CS156_Netflix

//Create by Yu Wu.

CS156b Course Project: Netflix Recommendation. This is the course project of CS156b Learning System, which is based on the famous Netflix competition for recommending movies to different users.

Group Name: **Netflix Official**

Group Member: **Haotian Sheng, Wen Gu, Diyi Liu, Sha Sha, Yu Wu**.

What we have implemented: (corresponding README.txt is included in the corresponding folders)

**RBM**, **SVD (SVD, SVD++, Time SVD++)**, **KNN (movie-movie)**, **Blending (Ridge Regression, XGBoost, Neural Network)**.
    
For a single model, Time SVD++ performs best with more than *6%* above water on Qual data. For Blending, we ensembled RBM, SVD, SVD++, time SVD++ and KNN and achieved a better performance with more than *7%* above water. Last but not least, we average the blended predictions, and also achieved a little improvement with about *7.5%* above water. More details can be found in our **Netflix_Official_Report_Final.pdf**.
