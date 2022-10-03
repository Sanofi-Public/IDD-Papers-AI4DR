import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def categorize(disp_proba, cat_proba, cat):
    if disp_proba > 0.5 :
        final_cat = 'Dispersion'
        proba = disp_proba
    elif disp_proba <= 0.5  and cat_proba> 0.9:
        final_cat = cat
        proba = cat_proba
    elif disp_proba <= 0.5  and cat_proba<= 0.9:
        final_cat = 'Low Probability'
        proba = cat_proba
    return [final_cat, proba]

def arrays_to_both_curves_images_w_title(pX_list, Y_list_norm, Y_list,title):
    pX_list = eval(pX_list.replace('nan','None'))
    Y_list = eval(Y_list.replace('nan','None'))
    Y_list_norm = eval(Y_list_norm.replace('nan','None'))
    assert len(pX_list) == len(Y_list) , 'Different number of replica for concentrations and inhibitions'
	# Function Figure (figure config [size point / scale] from 181112 param)
    mpl.rcParams["figure.dpi"] = 100
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (3.,2.0))                 #on charge une figure dans laquelle il y aura le graphique, de taille 1.5*1.5 en p50 en pixels
    for pX,Y in zip(pX_list,Y_list_norm):
        assert len(pX) == len(Y) , 'Different number of concentrations and inhibitions values'
        curr_line, = ax1.plot(pX,Y,'ko')                          #on affiche sur le même graphique les points des deux réplicats s'il y en a deux
        plt.setp(curr_line, markersize=3)                         #on réduit à 3 la taille des points
    for pX,Y in zip(pX_list,Y_list):
        assert len(pX) == len(Y) , 'Different number of concentrations and inhibitions values'
        curr_line, = ax2.plot(pX,Y,'ko')                          #on affiche sur le même graphique les points des deux réplicats s'il y en a deux
        plt.setp(curr_line, markersize=3)                         #on réduit à 3 la taille des points
    ax1.set_xticks([])                                          #on enlève l'affichage de l'échelle en abscisse
    ax2.set_xticks([])                                          #on enlève l'affichage de l'échelle en abscisse
    ax1.set_title('Normalized plot', fontsize=7 )
    ax2.set_title('Raw plot', fontsize=7)
    ax1.set_ylim(-50.0, 150.0)                                  #
    ax1.yaxis.set_tick_params(labelsize=5)                  # Taille des chiffre de l'échelle en Y
    ax2.yaxis.set_tick_params(labelsize=5)                  # Taille des chiffre de l'échelle en Y
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(pad=0.5)                               # pour recadrer la figure en prenant en compte les légendes / échelles / etc...

    img = fig.canvas.draw()
    return 

def arrays_to_both_curves_images_w_title_and_save(pX_list, Y_list_norm, Y_list,title,directory,file_basename):
    os.makedirs('./'+directory,exist_ok = True)
    pX_list = eval(pX_list.replace('nan','None'))
    Y_list = eval(Y_list.replace('nan','None'))
    Y_list_norm = eval(Y_list_norm.replace('nan','None'))
    assert len(pX_list) == len(Y_list) , 'Different number of replica for concentrations and inhibitions'
	# Function Figure (figure config [size point / scale] from 181112 param)
    mpl.rcParams["figure.dpi"] = 100
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (3.,2.0))                 #on charge une figure dans laquelle il y aura le graphique, de taille 1.5*1.5 en p50 en pixels
    for pX,Y in zip(pX_list,Y_list_norm):
        assert len(pX) == len(Y) , 'Different number of concentrations and inhibitions values'
        curr_line, = ax1.plot(pX,Y,'ko')                          #on affiche sur le même graphique les points des deux réplicats s'il y en a deux
        plt.setp(curr_line, markersize=3)                         #on réduit à 3 la taille des points
    for pX,Y in zip(pX_list,Y_list):
        assert len(pX) == len(Y) , 'Different number of concentrations and inhibitions values'
        curr_line, = ax2.plot(pX,Y,'ko')                          #on affiche sur le même graphique les points des deux réplicats s'il y en a deux
        plt.setp(curr_line, markersize=3)                         #on réduit à 3 la taille des points
    ax1.set_xticks([])                                          #on enlève l'affichage de l'échelle en abscisse
    ax2.set_xticks([])                                          #on enlève l'affichage de l'échelle en abscisse
    ax1.set_title('Normalized plot', fontsize=7 )
    ax2.set_title('Raw plot', fontsize=7)
    ax1.set_ylim(-50.0, 150.0)                                  #
    ax1.yaxis.set_tick_params(labelsize=5)                  # Taille des chiffre de l'échelle en Y
    ax2.yaxis.set_tick_params(labelsize=5)                  # Taille des chiffre de l'échelle en Y
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(pad=0.5)                               # pour recadrer la figure en prenant en compte les légendes / échelles / etc...

    #img = fig.canvas.draw()
    fig.savefig(directory+file_basename+'.png',bbox_inches="tight")

    return 

def summarize_save_triplicate_selection(df, directory, size=6):
    print('Dataframe size : '+ str(df.shape[0]) + ' showing the ' + str(size) + ' first entries')
    df.head()
    titles = ['{} - {} \nAI4DR : {} - Prob : {}'.format(w,x,y,round(z,2)) for w,x,y,z in zip(df['SAMPLE_ID'].head(size),df['ASSAY_OUTCOME'].head(size),df['Final_cat012'].head(size),df['Final_proba012'].head(size))]
    _ = [ arrays_to_both_curves_images_w_title_and_save(w, x, y, z, directory,v) for w,x,y,z,v in zip(df['pX_list'].head(size),df['Y_list'].head(size),df['Y_list_notr'].head(size),titles,df['SAMPLE_ID'].head(size))]
    return

def summarize_viz_triplicate_selection(df, size=6):
    print('Dataframe size : '+ str(df.shape[0]) + ' showing the ' + str(size) + ' first entries')
    df.head()
    titles = ['{} - {} \nAI4DR : {} - Prob : {}'.format(w,x,y,round(z,2)) for w,x,y,z in zip(df['SAMPLE_ID'].head(size),df['ASSAY_OUTCOME'].head(size),df['Final_cat012'].head(size),df['Final_proba012'].head(size))]
    _ = [ arrays_to_both_curves_images_w_title(w, x, y, z) for w,x,y,z in zip(df['pX_list'].head(size),df['Y_list'].head(size),df['Y_list_notr'].head(size),titles)]
    return
