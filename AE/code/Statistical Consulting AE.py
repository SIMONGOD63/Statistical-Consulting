#%% IMPORTS SECTION
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

#%% ###--------------- PREPROCESSING RESERVATION
reservations = pd.read_excel("01Data/reservation_251023_student1.xlsx")
# Rename columns
reservations.rename(columns={'statut_lvl1': 'cancelling',
                              'in_pre_saison': 'Av_1_an_res',
                              'type de sejour': 'Sej_spe'}, inplace=True)

#replace nan by a idk label
reservations["client_communaute"]  =reservations["client_communaute"].fillna("idk")
reservations["client_langue"]  =reservations["client_langue"].fillna("idk")
reservations["client_pays"]  =reservations["client_pays"].fillna("idk")


# Convert categorical variables to 'category' type

categorical_columns = ['bookingorigine', 'ae_staytype', 'Sej_spe', 'client_communaute',
                        'client_langue', 'client_pays', 'cancelling', 'Av_1_an_res', 'nbrnuits']
reservations[categorical_columns] = reservations[categorical_columns].astype('category')
del(categorical_columns)

# Convert numeric columns
numeric_columns = ['NbrAdo', 'NbrAdultes', 'NbrAnimaux', 'NbrBebe', 'NbrEnfants', 'nbrnuits']
reservations[numeric_columns] = reservations[numeric_columns].apply(pd.to_numeric, errors='coerce')
del(numeric_columns)

# Drop unnecessary columns
columns_to_drop = ['lastdate', 'enquête_typevacances'] #replicate of booking_date and too many NA
reservations.drop(columns=columns_to_drop, inplace=True)
del(columns_to_drop)

# Modify 'Sej_spe' columns
reservations['Sej_spe'] = reservations['Sej_spe'].str.replace(r'(.*)_sp', 'sp')

# Create a subset for 'AE' bookings
AE_data = reservations[reservations['bookingorigine'] == 'AE']


# Calculate the ratio of 'cancelling' in 'AE_data'
cancelling_ratio = (AE_data['cancelling'].value_counts() / len(AE_data['cancelling'])) * 100
print(cancelling_ratio)

#------------------------------------------ Create a subset for 'AE_data' without COVID

no_cov = deepcopy(AE_data[AE_data['debutsejour'] >= pd.to_datetime("2021-09-01")])
print((no_cov['cancelling'].value_counts() / len(no_cov['cancelling'])) * 100) #13.43 % annulation

#---Make sej spé a binary indicator
no_cov['Sej_spe'] = no_cov['Sej_spe'].apply(lambda x: 1 if 'sp' in x else 0)

#--- Make it binary if reservation made within the year
no_cov['Av_1_an_res'] = no_cov['Av_1_an_res'].apply(lambda x: 1 if x == 'InSeason' else 0)


#compute nbr days between booking and cancelling

no_cov['Can_timing'] = (no_cov['date_annulation'] - no_cov['booking_date']).dt.days
can_timg_summary = no_cov['Can_timing'].describe()
no_cov['Can_timing'] = no_cov['Can_timing'].fillna(-55) # fix nan values (they didnt cancel) with -55


# compute nbr days between sejour and cancelling (Days Before Sejour = d_b_s)
#no_cov['d_b_s'] = (no_cov['debutsejour'] - no_cov['date_annulation']).dt.days
#no_cov['d_b_s'] .describe()
#neg_val =no_cov[no_cov['d_b_s'] <= 0].index # awkward value

#just takes into account the month of the beginning of the sejour
no_cov['debutsejour'] = no_cov['debutsejour'].dt.month

no_cov.drop(columns = ["date_annulation","endsejour","booking_date"],inplace = True)


#%% DISCRETIZATION / "BINARIZATION"

# transform column animaux 
no_cov["Animaux"] = no_cov["NbrAnimaux"].apply(lambda x:0 if x ==0 else 1)
no_cov.drop(columns = ["NbrAnimaux"],inplace = True)

#no_cov['d_b_s'] .describe()
#no_cov['d_b_s2'] = pd.cut(no_cov['d_b_s'], [-170,0,23,64,156,800],labels = ["<0","0-23","23-64","64-156","+156"])
#no_cov.drop(columns = "d_b_s2", inplace = True)

#--- Create "cancelling" binary variable
no_cov["cancelling"] = no_cov["cancelling"].apply(lambda x: 1 if x == "Annulée" else 0)

#--- Tranform into categorical 
no_cov["debutsejour"] = no_cov["debutsejour"].astype("category")

###---- Discretize my continuous variables

#analyze their distribution
no_cov["booking_window"].describe() #makes no sense to have a negative value
no_cov['cat_BW'] = pd.cut(no_cov['booking_window'], [-3,47,104,200, 800], labels= ["0-47","47-104","104-200", "200+"])


no_cov["NbrAdultes"].describe() 
no_cov['cat_NAdults'] = pd.cut(no_cov['NbrAdultes'],[0,3,6,9,80], labels = ["1-3","3-6","6-9","9+"])
#test if the dicretization works as expected
test = deepcopy(reservations['NbrAdultes'])
test = pd.DataFrame(test)
test['cat_NAdults'] = pd.cut(reservations['NbrAdultes'],[0,3,6,9,80], labels = ["1-3","3-6","6-9","9+"])
#no problem with the discretization 
del(test)

no_cov['NbrAdo'].describe()
print(no_cov['NbrAdo'].isnull().sum())
no_cov["cat_NAdo"] = pd.cut(no_cov["NbrAdo"],[-1,1,55], labels = ["0","1+"])

no_cov['NbrBebe'].describe()
no_cov["cat_NBB"] = pd.cut(no_cov["NbrBebe"],[-1,1,55], labels = ["0","1+"])

print(can_timg_summary)
no_cov["cat_CancT"] = pd.cut(no_cov["Can_timing"],[-56,0,1,3,12,800],labels=["N-C","0d","1-3d","3-12d","12d+"],right = False)

no_cov['nbrnuits'].describe()
no_cov['N_nuits'] = pd.cut(no_cov['nbrnuits'],[0,2,3,4,130], labels=["2","3","4","4+"]) 

##------- standardize 

# =============================================================================
# scalerZ = StandardScaler()
# 
# no_cov["booking_window"] = scalerZ.fit_transform(no_cov['booking_window'].values.reshape(-1,1))
# mean(no_cov["booking_window"])
# to_stand = ["NbrAdultes","NbrAdo","NbrBebe","NbrEnfants","Can_timing"]
# for col in to_stand:
#    no_cov[col] = scalerZ.fit_transform(no_cov[col].values.reshape(-1,1))
# 
# =============================================================================

no_cov.drop(columns =['booking_window',"NbrAdultes","NbrAdo","NbrBebe","Can_timing","nbrnuits"],inplace = True) #cuz use discretized one

#transform binary into float32 types

no_cov[["cancelling","Animaux","Av_1_an_res","Sej_spe"]] = no_cov[["cancelling","Animaux","Av_1_an_res","Sej_spe"]].astype('float32')
#no_cov.drop(columns =['idbooking'], inplace = True) #cuz useless

#renaming some var
no_cov.rename(columns = {"Client_communaute":"CCommu","ae_staytype":"Stay_T","client_langue":"Client_L"}, inplace = True)


print(no_cov.dtypes)
print("Reservation DS ready and discretize")

#%%------ PREPROCESSING MAISON DATASET

maisons = pd.read_excel("01Data/maison_251023_student1.xlsx")
#either drop cap or cat_class cuz correlated/colinéaire
maisons.rename(columns = { 'maison_basicsegment_capacity' : 'cat_class',"date_premiere_mise_en_ligne":"1st online",
                          'equipement_enfants':'equip_e',"typeactionlocation":"contract","stat_maison_cote":"cote"}, inplace = True)

#supress cuz redondant or seem useless
maisons.drop(columns =  ['capacite','typebien', 'online_website',"avec_wellness","date_derniere_mise_hors_ligne","commune",'1st online',"has_concurrent" ], inplace = True)
maisons.drop(columns = [], inplace = True)
#transform columns
mapping = {0 :'no', 1:"yes", np.nan :'idk'}
maisons['exclusivité'] = maisons['exclusivité'].map(mapping)
#maisons['exclusivité'] = maisons['exclusivité'].astype('float64')

maisons['exclusivité']  = maisons['exclusivité'].astype("category") 
del(mapping)

# animaux as binary
maisons["animaux"] = maisons["animaux"].apply(lambda x:0 if x ==0 else 1)

#replacing nan values
maisons['recommandation']= maisons['recommandation'].fillna('idk')
maisons['ae_zone_name'] = maisons['ae_zone_name'].fillna('idk')
maisons['equip_e']= maisons['equip_e'].fillna('idk')
maisons['equipement'] = maisons['equipement'].fillna('idk')
#categorical col
cat_columns = ['cat_class','equipement',"equip_e",'recommandation',"pays", "SaisonsNom","ae_zone_name","code_postal"]
maisons[cat_columns] = maisons[cat_columns].astype("category")
del(cat_columns)

#check type
print(maisons.dtypes)

#maybe transform binary into float
int64 = maisons.select_dtypes(include ='int64').columns
maisons[int64] = maisons[int64].astype('float64')

#tranform nouveau into binary variable
maisons["nouveau"] = maisons['nouveau'].fillna(0)
maisons['nouveau'] = maisons["nouveau"].replace("nouveau",1)
maisons['nouveau'] = maisons['nouveau'].astype('float64')

# typeactionlocation into binary
maisons["contract"] = maisons['contract'].apply(lambda x:1 if x in [0,10] else 0)
maisons["contract"] = maisons['contract'].astype('float64')
print(maisons["contract"].value_counts()/len(maisons['contract']))
print("Only half of the houses are currently under contract")
#___________________________________________________________________________________________________
#--------DISCRETIZE CONTINUOUS VARIABLE
maisons["nbr_chambres"].describe()
maisons['N_rooms'] = pd.cut(maisons['nbr_chambres'],[-1,0,3,4,5,30], labels = ["0","1-3","4","5","5+"])

maisons["nbr_sdb"].describe()
maisons['N_sdb'] = pd.cut(maisons['nbr_sdb'],[-1,1,2,4,25], labels= ["0-1","2","3-4","4+"])


non_zero = maisons[maisons['cote'] !=0]['cote']
non_zero.describe() #min 6.1 mean = 9.13
maisons['cat_cote'] = pd.cut(maisons['cote'],[-1,8.9,9.2,9.5,11],labels =['0 - 8.9','8.9 - 9.2','9.2 - 9.5',"9.5+"],right = False)

maisons['nps_maison'].describe()
maisons['cat_nps'] = pd.cut(maisons['nps_maison'], [-1,8.0575, 8.745,9.2,11], labels = ['0-8','8- 8.75','8.75 - 9.2','9.2+'] )

to_drop =['nbr_sdb','nbr_chambres']
maisons.drop(columns = to_drop, inplace =True)
del(to_drop)
#---------scale some var
#to_scale = ['nbr_chambres', 'nbr_sdb']
#for col in to_scale:
#   maisons[col] = scalerZ.fit_transform(maisons[col].values.reshape(-1,1))

#rename filtre 
maisons.rename(columns = {"filtre_recommandation_animaux":"R_ani","filtre_recommandation_familleado":"Fam_Ado","filtre_recommandation_familleenfant":"Fam_enf","filtre_recommandation_groupeamis":"Group_Ami",
                                 "filtre_recommandation_couple":"couple","filtre_recommandation_qualiteprix":"qualiteprix","filtre_recommandation_velo":"velo",
                                 
                                 'filtre_detente_piscineexterieure':"P_ext","filtre_detente_sauna":"Sauna", "filtre_detente_piscine":"piscine","filtre_detente_hammam":"hammam","filtre_detente_jacuzzi":"jacuzzi",
                                 "filtre_detente_piscineinterieure":"P_int",
                                 
                                 "SaisonsNom":"Sais","ae_zone_name":"zone"}, inplace = True)


print(maisons.dtypes)
print("maison DS done and ready")


#%%-------------------------MERGE BOTH DATASETS---------------------------------------
merged_data = pd.merge(no_cov,maisons, on = "idmaison",how = 'inner')
print("data merged sucessfully")
#prop of contract == 1

print(merged_data['contract'].value_counts()/len(merged_data['contract']))
#should I discard those 5.9 percent of my data ?

# Create a subset where 'contract' is equal to 0
subset_contract_zero = merged_data[merged_data['contract'] == 0]

# Check for NaN values in the specified subset (replace 'your_columns' with actual column names)
nan_values_subset = subset_contract_zero[['cote', 'nps_maison']].isnull().any()

# Display NaN values information
print(nan_values_subset)

#proportion of nan
nan_values_nps_maison = subset_contract_zero['nps_maison'].isnull().mean()
nan_values_cote = subset_contract_zero['cote'].isnull().mean()

# Display the proportion of NaN values
print("Proportion of NaN values in 'nps_maison':", nan_values_nps_maison)
print("Proportion of NaN values in 'cote':", nan_values_cote)

#so I decide to remove this small proportion of my data (rows)
print(merged_data.shape)
merged_data = merged_data.dropna(subset=['nps_maison']) #lost 0.47 % of my data, its ok
print(merged_data.shape)

#----------------check again if there are still missing values
missing_values = merged_data.isnull()

# Identify the positions (row and column) of missing values
missing_positions = missing_values[missing_values].stack().index

# Display the positions of missing values
print("Positions of missing values:")
print(missing_positions)#no missing values

#-----remove the variable cote and nps_maison cuz discretize version
merged_data.drop(columns = ["cote","nps_maison","idbooking","idmaison"],inplace = True)
del(missing_values,int64,missing_positions, subset_contract_zero,nan_values_subset,nan_values_nps_maison,nan_values_cote)
print(merged_data.dtypes)

#%% Building the classic Auto encoder Neural network
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from copy import deepcopy

print(merged_data.dtypes) #float 32 and 64 idk if its a problem
# check number of dim add by each categorical variable
cat_columns = merged_data.select_dtypes(include=['category']).columns

# Get the number of unique levels for each categorical variable
levels_count = merged_data[cat_columns].nunique()

# Rank the categorical variables by the number of unique levels in decreasing order
ranked_levels_count = levels_count.sort_values(ascending=False)
print(ranked_levels_count)

merged_data['code_postal'].describe()#add 238 dimensions, i'll just drop it imo
merged_data['recommandation'].describe()# add 171 its fine


# IDK if necessary, maybe i can keep them
to_drop = ['code_postal','recommandation']
merged_data.drop(columns = to_drop, inplace = True)
del(to_drop)

#drop cuz uninformative and has same value accross the whole dataset
merged_data.drop(columns = 'bookingorigine',inplace = True) #cuz uninformative here

#check for variable who doesnt change over the DS => uninformative to classify indiv
unvar = []
for col in merged_data.columns:
    if merged_data[col].nunique() == 1:
        unvar.append(col)
print(unvar) #none of them
del(unvar)
# one hot encode the datas
def one_hot_encode(df):
    # Identify cat columns
    cat_columns = df.select_dtypes(include=['category']).columns

    # Apply one-hot encoding to string columns
    df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=False, dtype= float)

    return df_encoded

df_encoded = one_hot_encode(merged_data)
df_encoded.dtypes
# assign to all variables the float32 type
float64 =  df_encoded.select_dtypes(include='float64').columns
df_encoded[float64] = df_encoded[float64].astype('float32')
del(float64)


# error in VAE come from the fact that I have zeros in std_data
#lets find which columns give me zeros
std_data2 = np.std(df_encoded,axis=0)
idx02 = (std_data2 == 0) #getting the index of the columns for which std = 0
sum(idx02)
col0 = df_encoded.columns[idx02]
len(col0)
for colo in col0:
    print(df_encoded[colo].nunique())

#all of them only takes 1 values throughout the whole DS
# so I remove them
col0 = col0.tolist()
df_encoded.drop(columns = col0,inplace = True)

#-----------------------
# Converts it into an array using np.asarray()
ar_encoded = np.asarray(df_encoded, dtype = 'float')
test_0 = (np.std(ar_encoded,axis = 0)==0)
print("data ready for NN")

#%% ---------------------------VARIATIONAL AUTO ENCODER 
from keras_tuner import Hyperband

# Define the VAE model
std_data = np.std(ar_encoded,axis=0)
#check if values close to  0 within our std deviation
clean_0 = (std_data > 0.06)
std_data = std_data[clean_0]

#removing those variables from my DS
ar_encoded = ar_encoded[:,clean_0]
coding_size = 15
nc = ar_encoded.shape[1]

class VAE(keras.Model):
    def __init__(self, encoder, decoder,wgt, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.wgt     = wgt         #weights for the MSE

    def call(self, inputs):
       z_mean, z_log_var = self.encoder(inputs)
       z                 = self._sampling(z_mean, z_log_var)
       reconstruction    = self.decoder(z)
       loss              = self._VAE_loss(inputs,reconstruction, z_mean,z_log_var)
       self.add_loss(loss)
       return z, z_mean , z_log_var , reconstruction

    def _sampling(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z
    
    def _VAE_loss(self,inputs,outputs,z_mean,z_log_var):
        
# =============================================================================
#         bin_loss            = tf.losses.BinaryCrossentropy()
#         reconstruction_loss = bin_loss(inputs,outputs)
#         reconstruction_loss = tf.losses.MSE(inputs,outputs)
#         reconstruction_loss =  0.5*tf.reduce_sum(tf.square((inputs-outputs)/self.wgt)) 
# =============================================================================
        
        reconstruction_loss = 0.5 * tf.reduce_sum(tf.square(tf.divide((inputs - outputs), self.wgt))) #changed tf.divide
        kl_loss             = -0.5 * tf.reduce_sum(1 + (z_log_var) -
                               tf.square(z_mean) - tf.exp(z_log_var))
        return reconstruction_loss + kl_loss
            
# Now the loss function works

#%% HYPERBAND TUNING

def build_VAE1(hp):
    enco_inpt = layers.Input(shape=(nc,))
    deco_inpt = layers.Input(shape=(coding_size,))
    
    hle = None
    hld = None
    for i in range(hp.Int("n_layers",min_value = 1, max_value = 10, step = 1)):
        i +=1
        n_neurons = hp.Int("n_neurons", min_value = 10, max_value = 150,step =10)
        acti = hp.Choice('activation',["selu","relu"])
        if i == 1:
            #Hidden Layer Encoder
            hle = layers.Dense(units = n_neurons, activation =acti )(enco_inpt)
            #Hiddel Layer Decoder
            hld = layers.Dense(units = n_neurons, activation = acti)(deco_inpt)
        else:
            hle = layers.Dense(units = n_neurons, activation =acti )(hle)
            hld = layers.Dense(n_neurons, activation = acti)(hld)
    
    # coding size is the size of the reduced dimension
    z_mean = layers.Dense(coding_size,name="z_mean")(hle)
    z_log_var       = layers.Dense(coding_size,name="z_log_Var")(hle)
    decoder_out = layers.Dense(nc, activation ="sigmoid")(hld)
    
    #create var enco and deco
    var_encoder = keras.Model(inputs = [enco_inpt], outputs = [z_mean,z_log_var],name = "encoder")
    var_decoder = keras.Model(inputs = [deco_inpt], outputs = [decoder_out],name = "decoder")
    
    vae = VAE(var_encoder,var_decoder,std_data)
    vae.compile(optimizer=keras.optimizers.RMSprop(0.001))
    
    return vae

tuner2 = Hyperband(
    build_VAE1,
    objective ='loss',
    max_epochs = 500,
    hyperband_iterations = 5,
    directory = "tuning",
    project_name ="VAE4L.1",
    max_consecutive_failed_trials= 5)

log_dir2 = "logs_VAE4L_1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir=log_dir2, histogram_freq=1)
tuner2.search(ar_encoded,ar_encoded,
              epochs=300, 
             batch_size= len(ar_encoded), 
             callbacks=[tf.keras.callbacks.EarlyStopping('loss', patience=10),tensorboard_callback2])


#mod2 results
best_hps_val2 = tuner2.get_best_hyperparameters(num_trials=1)[0].values
best_hps2 = tuner2.get_best_hyperparameters(num_trials=1)[0]
model2 = tuner2.hypermodel.build(best_hps2)

history2 = model2.fit(ar_encoded,ar_encoded, epochs = 100, callbacks = tf.keras.callbacks.EarlyStopping('loss', patience=10)) #maybe increase the number of epochs here and add EarlyStop
val_loss_per_epoch2 = history2.history["loss"]
best_epoch2 = val_loss_per_epoch2.index(min(val_loss_per_epoch2)) 
print(best_epoch2)
print(val_loss_per_epoch2[best_epoch2])
hypermodel2 = tuner2.hypermodel.build(best_hps2)

# Retrain the model
hypermodel2.fit(ar_encoded, ar_encoded, epochs= best_epoch2 + 1,callbacks = tf.keras.callbacks.EarlyStopping('loss', patience=5))

#--- Get the encoder to compress my data
encoder2 = hypermodel2.get_layer("encoder")

#----Save the weights of my model

# =============================================================================
# hypermodel2.save_weights("hypermodel2_weights.h5")
# 
# ##---- Loading the weights
# hypermodel2.load_weights("hypermodel2_weights.h5")
# 
# =============================================================================
#--- Compress the data
df_compressed2,log_var2 = encoder2.predict(ar_encoded) #first one is the mean and that's what we are going to use

print("Data successfully compressed using Hyperband tuner2, congratz.")

# =============================================================================
# hypermodel2.save("best_HPM2.h5",save_fomrat = "tf")
# keras.saving.loaded_model = load_model("best_HPM2.h5", custom_objects = {'build_VAE1':build_VAE1,"class":VAE})
# #to launch tensorboard
# import subprocess
# log_dir = "logs_new1/fit"
# command = ["tensorboard","--logdir",log_dir]
# subprocess.run(command, check = True)
# =============================================================================

#%%--------------------- K MEANS PART
# then do a K means
n_clust = 10 #number of profile

km2 = KMeans( init = 'k-means++', # se init = "k-means++'
            n_clusters = n_clust,
            n_init = 20,
            max_iter = 300,
            random_state=4243)
# try the the diff between the two initialization


# so from now on lets use km2
## getting the optimal number of clusters
def opt_clust(DS):
    dist = []
    K_range = range(1,15)
    for k in K_range :
        kmeans = KMeans(init = 'k-means++',
                    n_clusters = k,
                    n_init = 20,
                    max_iter = 300,
                    random_state=4243)
        kmeans.fit(DS)
        dist.append(kmeans.inertia_)
    #plotting
    plt.plot(K_range, dist, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

###----------------- contructing the table  

def make_tab(km,n_clust,df,DS):
    km.fit(df)
    print("Inertia : ",km.inertia_)
    
    # Indicates to which class each point of the DS correspond
    X_lab_Clust = km.labels_  
    
    #creating the table
    names = DS.columns.tolist()
    tab = np.zeros(shape = (n_clust,len(names)))
    tab = pd.DataFrame(data = tab, columns = names)
    tab['Can_Prop'] = 0
    tab.astype(float)
    
    #filling the dataframe
    for k in range(0,n_clust):
        #get the index of the data points that belongs to the current class
        idx = (X_lab_Clust == k) 
        
        # computing the prop of cancelling within the class
        tab.loc[k,"Can_Prop"] = round((sum(DS['cancelling'][idx])/ sum(idx))*100,2) # DS if merged_data for classicale talbe
        #computing the number of ind per cluster
        tab.loc[k,"prop_ind"] = round((sum(idx)/len(DS))*100,2)
        for col in tab.columns[:56]:
            tab.loc[k,col] = Counter(DS[col][idx]).most_common(1)[0][0]
    return tab


##

opt_clust(df_compressed2)
### CREATING THE TABLE
table2 = make_tab(km2, 10, df_compressed2,merged_data) #work on table 2 cuz model 2 has the smallest loss
table2.to_csv('kmeans_table.csv',sep=",")

Original_table = deepcopy(table2)

#lets compare kmeans2_1 and kmeans2_0
table2_0= pd.read_csv("kmeans2_0.csv",sep=",")


#%%Analysis of the proportion within our data

#sej spe
print((merged_data["Sej_spe"].value_counts()/len(merged_data))*100) #88% of sej_spe

#stay type
print((merged_data["ae_staytype"].value_counts()/len(merged_data))*100) #51% weekends

#
print((merged_data["ae_zone_name"].value_counts()/len(merged_data))*100)

#N enfant
print((merged_data["NbrEnfants"].value_counts()/len(merged_data))*100) #62 % sans enfants

#langue 
print((merged_data["client_langue"].value_counts()/len(merged_data))*100) #80% NL

#pays 
print((merged_data["client_pays"].value_counts()/len(merged_data))*100)  #72% BE

#Animaux
print((merged_data["Animaux"].value_counts()/len(merged_data))*100)  #77% without animals

print((merged_data["cat_NAdo"].value_counts()/len(merged_data))*100)  # 87 % zero

print((merged_data["cat_NBB"].value_counts()/len(merged_data))*100)  # 90 %

print((merged_data["cat_CancT"].value_counts()/len(merged_data))*100) # categories of cancelation

print((merged_data["exclusivité"].value_counts()/len(merged_data))*100) # majority of no 67%

#equip_enfant et equipement
print((merged_data["equip_e"].value_counts()/len(merged_data))*100) 
print((merged_data["equipement"].value_counts()/len(merged_data))*100) 

#flitre 
print((merged_data["filtre_detente_hammam"].value_counts()/len(merged_data))*100) #97% no
print((merged_data["filtre_detente_jacuzzi"].value_counts()/len(merged_data))*100) # 76 % no

#nouveau reco promo
print((merged_data["filtre_nouveau_nouveau"].value_counts()/len(merged_data))*100)  #99% no
print((merged_data["filtre_promotion_promo"].value_counts()/len(merged_data))*100) # 73%
print((merged_data["filtre_recommandation_animaux"].value_counts()/len(merged_data))*100)  #70%

print((merged_data["filtre_recommandation_velo"].value_counts()/len(merged_data))*100) #77% no
print((merged_data["filtre_recommandation_pmr"].value_counts()/len(merged_data))*100) # 90 % no
print((merged_data["nouveau"].value_counts()/len(merged_data))*100) 
#maisons
print((merged_data["filtre_typebien_maison"].value_counts()/len(merged_data))*100) #68% de maisons
#priorité
print((merged_data["priorite"].value_counts()/len(merged_data))*100) #86%

#pays
print((merged_data["pays"].value_counts()/len(merged_data))*100) #99%

print((merged_data["contract"].value_counts()/len(merged_data))*100) #94%

#%%----------------------------------------------CLEANING THE TABLE
# drop the uninformative variables (the  variables staying constant)
unvar = []
for col in table2.columns:
    if table2[col].nunique() == 1:
        unvar.append(col)
print(unvar)


# only keep informative looking information 
table2.drop(columns=unvar, inplace = True)
table2.drop(columns = ['equip_e',"equipement"], inplace = True)



# on arrondi les variables numériques clé
table2['Can_Prop'] = round((Original_table['Can_Prop']).astype(int),0)
table2['prop_ind'] = round((Original_table['prop_ind']).astype(float),1)

#download table
table2.to_csv('table_pres.csv',sep=",",index = False,decimal=".")



###---------------------------------------- EXPORTING the table
import dataframe_image as dfi
dfi.export(table2,"final_test.png")

#%%############################################################################################################################################################################################
#
#--------------------------------------------------------------- creating class wihtin cancelling only
#
############################################################################################################################################################################################

#--- Setup
can_ds = merged_data[merged_data["cancelling" ] == 1]
can_encoded = one_hot_encode(can_ds)
ar_can = np.asarray(can_encoded,dtype = "float")
std_data3 = np.std(can_encoded,axis=0)


prob2 = []
for i in range(len(std_data3)):
    if std_data3[i] <= 0.01:
        prob2.append((i,std_data3[i]))
print(len(prob2))

idx03 = (std_data3 <= 0.01) #getting the index of the columns for which std = 0
sum(idx03) # I have col which equals std = 0
# it pause problem
col03 = can_encoded.columns[idx03]
len(col03)
col03 = col03.tolist()


can_encoded.drop(columns = col03,inplace = True)

ar_can = np.asarray(can_encoded,dtype = "float")
#tuning part
#______________________________________________________________________________________________________________
#those 3 has to be set before using an VAE
nc = ar_can.shape[1]
coding_size = 15
std_data = np.std(ar_can,axis=0)

###########

tuner3 = Hyperband(
    build_VAE1,##just try using VAE1
    objective ='loss',
    max_epochs = 500,
    hyperband_iterations = 5,
    directory = "tuning",
    project_name ="VAE4L.tuner3",
    max_consecutive_failed_trials= 5)

log_dir3 = "logs_VAE4L_t3/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback3 = tf.keras.callbacks.TensorBoard(log_dir=log_dir3, histogram_freq=1)
tuner3.search(ar_can,ar_can,
              epochs=300, 
             batch_size= len(ar_can),
             callbacks=[tf.keras.callbacks.EarlyStopping('loss', patience=10),tensorboard_callback3])


#mod2 results
best_hps_val3 = tuner3.get_best_hyperparameters(num_trials=1)[0].values

best_hps3 = tuner3.get_best_hyperparameters(num_trials=1)[0]
model3 = tuner3.hypermodel.build(best_hps3)

history3 = model3.fit(ar_can,ar_can, epochs = 100, callbacks = tf.keras.callbacks.EarlyStopping('loss', patience=10)) #maybe increase the number of epochs here
val_loss_per_epoch3 = history3.history["loss"]
best_epoch3 = val_loss_per_epoch3.index(min(val_loss_per_epoch3)) 
print(best_epoch3)
print(val_loss_per_epoch3[best_epoch3]  )
hypermodel3 = tuner3.hypermodel.build(best_hps3)

# Retrain the model
hypermodel3.fit(ar_can, ar_can, epochs= best_epoch3 + 1, callbacks =tf.keras.callbacks.EarlyStopping('loss', patience=2))
#get the architecture
hypermodel3.get_layer("encoder").summary()
hypermodel3.get_layer("decoder").summary()
#get the encoder to compress my data
encoder3 = hypermodel3.get_layer("encoder")

#get compressed data
df_compressed3,log_var3 = encoder3.predict(ar_can)

#%%
# =============================================================================
# #-----------------------TABLE
# =============================================================================

opt_clust(df_compressed3)
n_clust3 = 10
table3 = make_tab(km2, n_clust3, df_compressed3,can_ds)

#### analyze and clean the table 
unvar = []
for col in table3.columns:
    if table3[col].nunique() == 1:
        unvar.append(col)
print(unvar)
unvar.remove("Av_1_an_res") # we keed those because informative
to_drop = ["equip_e","equipement","filtre_promotion_promo"]
table3.drop(columns= to_drop, inplace = True)
table3.drop(columns = unvar, inplace  = True)
print("table3 done !")
del(to_drop)
del(unvar)

#download the table
dfi.export(table3,"Table2_redu.png")
