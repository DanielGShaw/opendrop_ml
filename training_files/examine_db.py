import optuna
import os

files = []
for file in os.listdir():
    if file[-3:]=='.db':
        files.append(file)

print('db study files found: ', files)

for file in files:
    study_name = file[:-3]
    storage="sqlite:///{}.db".format(study_name)
    print(storage)
    study = optuna.load_study(study_name=study_name, storage=storage)

    #optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])

    if 0: #show figures
        fig = optuna.visualization.plot_contour(study, params=["model_width", "batch_size"])
        fig.show()
    
        fig = optuna.visualization.plot_contour(study, params=["model_width", "learning_rate"])
        fig.show()
    
        fig = optuna.visualization.plot_contour(study, params=["learning_rate", "batch_size"])
        fig.show()
    
        fig = optuna.visualization.plot_slice(study, params=["model_width", "batch_size", "learning_rate", "es_patience"])
        fig.show()
    
        fig = optuna.visualization.plot_param_importances(
            study, target=lambda t: t.values[0], target_name="score")
        fig.show()

        fig = optuna.visualization.plot_timeline(study)
        fig.show()

    outputs = {}
    for i, trial in enumerate(study.get_trials()):
        if i==0:
            print(trial)
            print()
        #print(trial.values)
        #print(trial.params)
        if trial.values is not None:
            outputs[trial.values[0]]=trial.params

    keys = sorted(outputs.keys())

    for key in keys:
        print(key)
        print(outputs[key])
        print()
