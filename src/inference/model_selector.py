from src.inference.gector import GectorModel


MODEL_PATHS = {
    'gector-roberta' : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/roberta_1_gectorv2.th',
    'gector-bert'    : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/bert_0_gectorv2.th',
    'gector-xlnet'   : '/scratches/dialfs/alta/vr313/GEC/spoken-gec-combination/experiments/trained_models/xlnet_0_gectorv2.th'
}

def select_model(args):
    mname = args.model_name
    model_path = MODEL_PATHS[mname]
    breakpoint()
    if 'gector' in mname:
        transformer_model = mname.split('-')[-1]
        return GectorModel(args, transformer_model, model_path)