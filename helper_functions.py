def get_logs(path):       

    path = [(path + "/".join([str(int(p["trackID"])), p["date"].strftime('%Y-%m-%d'), str(p['time/log'].hour)]) , p["name"]) 
            for i,p in gt.iterrows()]

    logs = [(["/".join([p, f]) for f in os.listdir(p)], l) for p,l in path]
    return logs


def get_start_end_log(log):
    logs, name = log
    start = []
    stop = []
    between  = []
    I_empty = []
    
    p = re.compile("([\d-]+),\s+([\d:]+),\s(\w+),\s+(\d+),\s+(\d+),\s([\w ]+),\s([\d.-e]+),\s+([\[\d\.\],e -]+)")
    
    for I,l in enumerate(logs):
        total_l = sum(1 for line in open(l))
        sta = subprocess.check_output(['head', '-1', l]).strip().decode('UTF-8')
        sto = subprocess.check_output(['tail', '-1', l]).strip().decode('UTF-8')
        if total_l > 2:
            
            pat = "2," + str(total_l-1) + "p;" + str(total_l) + "q"
            bet = subprocess.check_output(['sed', '-n', pat, l]).decode('UTF-8')#.strip().decode('UTF-8').split("\n") #sed -n '16224,16482p;16483q' filename)
            between.append(re.findall(p, bet))            
        elif total_l == 2:
            ext = re.compile("([0-9A-Za-z]+)_([\d_]+)([A-Za-z]+)(\d+)_\w+_(\d+).png")
            PAT = "/".join(["smart_surveillance/FaceRecognition/FaceNet_input/Eingang/out_of_interest_area/trackID_" + l.split("_")[-1].split(".")[0]])
            pat_list = os.listdir(PAT)
            BETWEEN = []
            
            START_da = datetime.strftime(datetime.strptime(sta.split(", ")[0],"%Y-%m-%d"),"%Y-%m-%d")
            STOP_da  = datetime.strftime(datetime.strptime(sto.split(", ")[0],"%Y-%m-%d"),"%Y-%m-%d")

            START_ti = datetime.strftime(datetime.strptime(sta.split(", ")[1],"%H:%M:%S"),"%H:%M:%S")
            STOP_ti  = datetime.strftime(datetime.strptime(sto.split(", ")[1],"%H:%M:%S"),"%H:%M:%S")

            for P in pat_list:
                da,ti,so,fr,tr = [ext.search(P).group(i) for i in range(1,6)]
                da = datetime.strftime(datetime.strptime(da, "%Y%B%d"), "%Y-%m-%d")
                ti = datetime.strftime(datetime.strptime(ti, "%H_%M_%S_"), "%H:%M:%S")
                
                if (da == START_da) | (da ==STOP_da):
                    if (ti >= START_ti) & (ti<=STOP_ti):
                        BETWEEN.append((da,ti,so,fr,tr,None,None,None))
                        

            BETWEEN = sorted(BETWEEN, key=itemgetter(0,1,3))
            
            
            if BETWEEN == []:
                try:
                    between.append([(START_da, START_ti, "Eingang", '0', tr, None, None, None)])
                except Exception as e:
                    print(e)
                    between.append([(START_da, START_ti, "Eingang", '0', '0', None, None, None)])
                
                I_empty.append(I)
            else:
                between.append(BETWEEN)

                
        start.append(sta.split(", "))
        stop.append(sto.split(", "))

    start = sorted(start, key=itemgetter(0,1))
    stop = sorted(stop, key=itemgetter(0,1))
    

    try:
        between = sorted(between, key=itemgetter(0,3))
    except Exception as e:
        print(e)
        print(between)
    
#     for i in I_list:
#         between[i] = [between[i][0]]
    for i in I_empty:
        between[i] = []

    return name, start, stop, between

def get_pics(path, bet, name):
    if name != "out_of_area":
        path = [(path + p[2] + "/in_of_interest_area/" + "trackID_" + str(int(p[4])), datetime.strftime(datetime.strptime(p[0], '%Y-%m-%d'), '%Y%B%d_') )
        if p != [] 
        else (path + p[2] + "/out_of_interest_area/" + "trackID_" + str(int(p[4])), datetime.strftime(datetime.strptime(p[0], '%Y-%m-%d'), '%Y%B%d_') )
        for log in bet for p in log]
    else:
        path = [(path + p[2] + "/out_of_interest_area/" + "trackID_" + str(int(p[4])), datetime.strftime(datetime.strptime(p[0], '%Y-%m-%d'), '%Y%B%d_'))
         for log in bet for p in log]
        

#     print(path)
    path = list(set(path))
    files = [["/".join([p] + [f]) for f in sorted(os.listdir(p)) if f.startswith(d)] for p,d in path]
    
    return files[0]


# # Converting links to html tags
def path_to_image_html(path):
    return '<img src="'+ path + '" width="40" >'

def rep(x):
    return [x[0]] * x[1]

def load_database(database="/mnt/golem/frodo/clusteredFaces_lower_resolution/final_low_resoloution_FaceDB.json"):
    db = json.loads(open(database).read())
    res = list(db.keys())
    
    db2 = {res[0]: {k: np.vstack(v["encodings"]) for k,v in db[res[0]].items()}}
    len_db = {res[0]: list(map(len, list(db2[res[0]].values())))}
    nest = {res[0]: list(map(rep, zip(list(db2[res[0]].keys()), len_db[res[0]])))}
    names = {res[0]: list(chain(*nest[res[0]]))}

    for r in res[1:]:
        db2.update({r: {k: np.vstack(v["encodings"]) for k,v in db[r].items()}})
        len_db.update({r: list(map(len, list(db2[r].values())))})
        nest.update({r: list(map(rep, zip(list(db2[r].keys()), len_db[r])))})
        names.update({r: list(chain(*nest[r]))})
    
    arr = {k: (np.vstack(list(v.values())), names[k]) for k,v in db2.items()}
    
    return arr



def split_data(arr, resolution="64", proportion_class=80, exclude_class ="Maureen_Leber", proportion_known = 0.05,
              seed=42):
    # How many classes result in 80% of all classes
    random.seed(seed)
    X_tot,y_tot = arr[resolution]
#     print(X_tot.shape, len(y_tot))
    thr = int(proportion_class * len(Counter(y_tot)) /100)

    X_train = np.zeros(128)
    X_test = np.zeros(128)
    y_train = []
    y_test = []

    n = 1
    freq_classes = dict(Counter(y_tot))
    tr = sample(freq_classes.keys(),thr)
    te = list(set(y_tot)-set(tr))
    for c,v in freq_classes.items():    
        ind = y_tot.index(c)
        if c == exclude_class:
            continue
        elif c in tr:
            y_train += y_tot[ind:ind+v]
            X_train = np.vstack((X_train, X_tot[ind:ind+v,:]))
        elif c in te:
            y_test += y_tot[ind:ind+v]
            X_test = np.vstack((X_test, X_tot[ind:ind+v,:]))

    X_train = np.delete(X_train, (0), axis=0)
    X_test = np.delete(X_test, (0), axis=0)
#     print(X_train.shape, len(y_train), X_test.shape, len(y_test))

    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_train, y_train, test_size=proportion_known, random_state=42)

    X_test = np.vstack((X_test, X_sub_test))
    y_test += y_sub_test
#     print(X_train.shape, len(y_train), X_test.shape, len(y_test))
    return X_train, y_train, X_test, y_test, y_sub_test

def performance(pred, pr=False):
    act_pos = Counter(elem[0] for elem in pred if (elem[0] != elem[3]) & (elem[2] == "unknown"))
    TP = act_pos["unknown"]
    FN = sum(act_pos.values())-TP
#     print(TP, FN)
    act_neg = Counter(elem[0] for elem in pred if (elem[2] == "known"))
    FP = act_neg["unknown"]
    TN = sum(act_neg.values())-FP
#     print(FP, TN)
    try:
        precision = TP / (TP+FP)
    except Exception as e:
        print(e)
        precision = 0
    try:
        recall = TP / (TP+FN)
    except Exception as e:
        print(e)
        recall = 0
    try:
        accuracy = (TP+TN) / (TP+TN+FP+FN)
    except Exception as e:
        print(e)
        accuracy = 0
    try:
        f1 = 2* (precision*recall)/(precision+recall)
    except Exception as e:
        print(e)
        f1 = 0
    if pr:
        new_line = os.linesep
        print(f"precision: {precision:.2f}{new_line}recall: {recall:.2f}{new_line}accuracy: {accuracy:.2f}{new_line}f1: {f1:.2f}")
    return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1}

def clf_predict(X_test,y_test,y_sub_test, clf = ["knn", "euclidean", "random", "sgd"]):
    pred_thr = {}
    loop = np.linspace(start=0, stop=1, num=17)[np.arange(0,17,2)].tolist()
    

    knn = KNeighborsClassifier(weights="distance", n_neighbors=3)
    knn.fit(X_train, y_train)
    min_dist_ecl = 100
    
    SGD = SGDClassifier(max_iter=1000, alpha=0.01, class_weight = "balanced", loss='modified_huber' )
    SGD.fit(X_train, y_train)

    for r,label in zip(X_test, y_test):    
        if label in y_sub_test:
            mode = "known"
        else:
            mode = "unknown"

        ident_knn, dist_knn, status = knn.predict(r.reshape(1,128))
        orig_ident_knn = ident_knn.copy()
        n,w = dist_knn[0]
        dist_knn = list(zip(n,w))
        dist_ind = [x for x, y in enumerate(dist_knn) if y[0] ==  ident_knn]

        min_dist_knn = 1/np.array(dist_knn)[dist_ind][:,1].astype(float).mean()
        
        for thr in tqdm(loop):
            if thr == 0.0:
                thr = 1/100000
                
            if min_dist_knn < 1/thr:
                ident_knn = "unknown"
            else:
                ident_knn = orig_ident_knn[0]
            if "knn" in pred_thr.keys():
                if thr in pred_thr["knn"].keys():
                    pred_thr["knn"][thr] +=  [(ident_knn, min_dist_knn, mode, label)]
                else:
                    pred_thr["knn"][thr] =  [(ident_knn, min_dist_knn, mode, label)]                    
            else:
                pred_thr["knn"] = {thr: [(ident_knn, min_dist_knn, mode, label)]}
                
            if "euclidean" in clf:
                dist_ecl = np.apply_along_axis(np.linalg.norm, 1, (X_train - r.reshape(1,128)))
                ind = np.argmin(dist_ecl)
                if dist_ecl[ind] < min_dist_ecl:
                    min_dist_ecl = dist_ecl[ind]
                    ident_ecl = y_train[ind]
                if min_dist_ecl < thr:
                    ident_ecl = "unknown"
                else:
                    ident_ecl = ident_ecl
                if "ecl" in pred_thr.keys():
                    if thr in pred_thr["ecl"].keys():                    
                        pred_thr["ecl"][thr] +=  [(ident_ecl, min_dist_ecl, mode, label)]
                    else:
                        pred_thr["ecl"][thr] =  [(ident_ecl, min_dist_ecl, mode, label)]
                else:
                    pred_thr["ecl"] = {thr: [(ident_ecl, min_dist_ecl, mode, label)]}
                    
                min_dist_ecl = 100
                    
            if "sgd" in clf:        
                proba = SGD.predict_proba(r.reshape(1,128))         

                #thr = 1/len(proba[0,:])*10
#                 print(proba, thr)

                ind = np.where(proba>=thr)

                if ind[1] != []:
                    ix = np.argmax(proba[0,:])
                    ident_sgd = SGD.classes_[ix]
                    min_dist_sgd = proba[0,ix]
                else:
                    ident_sgd = "unknown"
                    min_dist_sgd = 0

                if "sgd" in pred_thr.keys():
                    if thr in pred_thr["sgd"].keys():                    
                        pred_thr["sgd"][thr] +=  [(ident_sgd, min_dist_sgd, mode, label)]
                    else:
                        pred_thr["sgd"][thr] =  [(ident_sgd, min_dist_sgd, mode, label)]
                else:
                    pred_thr["sgd"] = {thr: [(ident_sgd, min_dist_sgd, mode, label)]}
                    
                    
            if "random" in clf:        
                proba_rand = np.array([random.uniform(0, 1) for i in range(len(y_test))])
#                 ident_rand = sample(list(set(y_test)), 1) 
#                 print(proba_rand)
                ind = np.where(proba_rand>=thr)
#                 print(ind)
                if ind != []:
                    ix = np.argmax(proba_rand)
                    ident_rand = list(y_test)[ix]
                    min_dist_rand = proba_rand[ix]
                else:
                    ident_rand = "unknown"
                    min_dist_rand = proba_rand[ix]

                if "rand" in pred_thr.keys():
                    if thr in pred_thr["rand"].keys():                    
                        pred_thr["rand"][thr] +=  [(ident_rand, min_dist_rand, mode, label)]
                    else:
                        pred_thr["rand"][thr] =  [(ident_rand, min_dist_rand, mode, label)]
                else:
                    pred_thr["rand"] = {thr: [(ident_rand, min_dist_rand, mode, label)]}
                
                    
    return pred_thr

