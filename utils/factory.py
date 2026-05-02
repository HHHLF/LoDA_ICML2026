def get_model(model_name, args):
    name = model_name.lower()
    if name == 'macil':
        from methods.macil import Learner
    if name == 'dlora':
        from methods.dlora import Learner
    return Learner(args)

