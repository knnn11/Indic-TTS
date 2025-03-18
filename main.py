import os
import sys
from inference.src.utils.text import str2bool

import argparse
import os
import string

import numpy as np
import pandas as pd
import torch

from argparse import Namespace
from torch.utils.data import DataLoader
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.models.align_tts import AlignTTS
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_checkpoint
from tqdm.auto import tqdm

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for acoustic / e2e TTS model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='indictts', choices=['ljspeech', 'indictts', 'googletts'])
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'te', 'kn', 'ml', 'hi', 'mr', 'bn', 'gu', 'or', 'as', 'raj', 'mni', 'brx', 'all'])
    parser.add_argument('--dataset_path', default='/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/{}/{}', type=str) # dataset_name, language #CHANGE
    parser.add_argument('--speaker', default='all') # eg. all, male, female, ...
    parser.add_argument('--use_phonemes', default=False, type=str2bool)
    parser.add_argument('--phoneme_language', default='en-us', choices=['en-us'])
    parser.add_argument('--add_blank', default=False, type=str2bool)
    parser.add_argument('--text_cleaner', default='multilingual_cleaners', choices=['multilingual_cleaners'])
    parser.add_argument('--eval_split_size', default=0.01)
    parser.add_argument('--min_audio_len', default=1)
    parser.add_argument('--max_audio_len', default=float("inf")) # 20*22050
    parser.add_argument('--min_text_len', default=1)
    parser.add_argument('--max_text_len', default=float("inf")) # 400
    parser.add_argument('--audio_config', default='without_norm', choices=['without_norm', 'with_norm'])

    # model parameters
    parser.add_argument('--model', default='glowtts', choices=['glowtts', 'vits', 'fastpitch', 'tacotron2', 'aligntts'])
    parser.add_argument('--hidden_channels', default=512, type=int)
    parser.add_argument('--use_speaker_embedding', default=True, type=str2bool)
    parser.add_argument('--use_d_vector_file', default=False, type=str2bool)
    parser.add_argument('--d_vector_file', default="", type=str)
    parser.add_argument('--d_vector_dim', default=512, type=int)
    parser.add_argument('--speaker_encoder_model_path', default='', type=str) 
    parser.add_argument('--speaker_encoder_config_path', default='', type=str) 
    parser.add_argument('--use_speaker_encoder_as_loss', default=False, type=str2bool) # only supported in vits, fastpitch
    parser.add_argument('--use_ssim_loss', default=False, type=str2bool) # only supported in fastpitch
    parser.add_argument('--vocoder_path', default=None, type=str) # external vocoder for speaker encoder loss in fastpitch
    parser.add_argument('--vocoder_config_path', default=None, type=str)  # external vocoder for speaker encoder loss in fastpitch
    parser.add_argument('--use_style_encoder', default=False, type=str2bool)
    parser.add_argument('--use_aligner', default=True, type=str2bool) # for fastspeech, fastpitch
    parser.add_argument('--use_separate_optimizers', default=False, type=str2bool) # for aligner in fastspeech, fastpitch
    parser.add_argument('--use_pre_computed_alignments', default=False, type=str2bool) # for fastspeech, fastpitch
    parser.add_argument('--pretrained_checkpoint_path', default=None, type=str) # to load pretrained weights
    parser.add_argument('--attention_mask_model_path', default='output/store/ta/fastpitch/best_model.pth', type=str) # set if use_aligner==False and use_pre_computed_alignments==False #CHANGE
    parser.add_argument('--attention_mask_config_path', default='output/store/ta/fastpitch/config.json', type=str) # set if use_aligner==False and use_pre_computed_alignments==False #CHANGE
    parser.add_argument('--attention_mask_meta_file_name', default='meta_file_attn_mask.txt', type=str) # dataset_name, language # set if use_aligner==False #CHANGE

    # training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--aligner_epochs', default=1000, type=int) # For FastPitch
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_eval', default=8, type=int)
    parser.add_argument('--batch_group_size', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_workers_eval', default=8, type=int)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)
    parser.add_argument('--compute_input_seq_cache', default=False, type=str2bool)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_scheduler', default='NoamLR', choices=['NoamLR', 'StepLR', 'LinearLR', 'CyclicLR', 'NoamLRStepConstant', 'NoamLRStepDecay'])
    parser.add_argument('--lr_scheduler_warmup_steps', default=4000, type=int) # NoamLR
    parser.add_argument('--lr_scheduler_step_size', default=500, type=int) # StepLR
    parser.add_argument('--lr_scheduler_threshold_step', default=500, type=int) # NoamLRStep+
    parser.add_argument('--lr_scheduler_aligner', default='NoamLR', choices=['NoamLR', 'StepLR', 'LinearLR', 'CyclicLR', 'NoamLRStepConstant', 'NoamLRStepDecay'])
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float) # StepLR, LinearLR, CyclicLR

    # training - logging parameters 
    parser.add_argument('--run_description', default='None', type=str)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--test_delay_epochs', default=0, type=int)   
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--save_step', default=10000, type=int)
    parser.add_argument('--save_n_checkpoints', default=1, type=int)
    parser.add_argument('--save_best_after', default=10000, type=int)
    parser.add_argument('--target_loss', default=None)
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--run_eval', default=True, type=str2bool)
    
    # distributed training parameters
    parser.add_argument('--port', default=54321, type=int)
    parser.add_argument('--continue_path', default="", type=str)
    parser.add_argument('--restore_path', default="", type=str)
    parser.add_argument('--group_id', default="", type=str)
    parser.add_argument('--use_ddp', default=True, type=bool)
    parser.add_argument('--rank', default=0, type=int)
    #parser.add_argument('--gpus', default='0', type=str)

    # vits
    parser.add_argument('--use_sdp', default=True, type=str2bool)

    return parser


def formatter_indictts(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs-22k", cols[0] + ".wav")
            text = cols[1].strip()
            speaker_name = cols[2].strip()
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


def filter_speaker(samples, speaker):
    if speaker == 'all':
        return samples
    samples = [sample for sample in samples if sample['speaker_name']==speaker]
    return samples


def get_lang_chars(language):
    if language == 'ta':
        lang_chars_df = pd.read_csv('chars/Characters-Tamil.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = ['ௗ', 'ஹ', 'ஜ', 'ஸ', 'ஷ']
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'hi':
        lang_chars_df = pd.read_csv('chars/Characters-Hindi.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = []
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'en':
        lang_chars = string.ascii_lowercase

    return lang_chars


def get_test_sentences(language):
    if language == 'ta':
        test_sentences = [
                "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்[...]",
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின்[...]"
            ]

    elif language == 'en':
        test_sentences = [
                "Brazilian police say a suspect has confessed to burying the bodies of missing British journalist Dom Phillips and indigenous expert Bruno Pereira.",
                "Protests have erupted in India over a new reform scheme to hire soldiers for a fixed term for the armed forces",
            ]
        
    elif language == 'mr':
        test_sentences = [
                "मविआ सरकार अल्पमतात आल्यानंतर अनेक निर्णय घेतले: मुख्यमंत्री एकना�[...]",
                "वर्ध्यात भदाडी नदीच्या पुलावर कार डिव्हायडरला धडकून भीषण अपघात, दो[...]"
            ]

    elif language == 'as':
        test_sentences = [
                "দেউতাই উইলত স্পষ্টকৈ সেইখিনি মোৰ নামত লিখি দি গৈছে",
                "গতিকে শিক্ষাৰ বাবেও এনে এক পূৰ্ব প্ৰস্তুত পৰি‌ৱেশ এটাত",
            ]

    elif language == 'bn':
        test_sentences = [
                "লোডশেডিংয়ের কল্যাণে পুজোর দুসপ্তাহ আগে কেনাকাটার মাহেন্দ্রক্ষণে,[...]",
                "এক চন্দরা নির্দোষ হইয়াও, আইনের আপাত নিশ্ছিদ্র জালে পড়িয়া প্রাণ দিয়��[...]"
            ]

    elif language == 'brx':
        test_sentences = [
                "गावनि गोजाम गामি नवथिखौ हरखाब नागारनানै गोदान हादानाव गावखौ दिदोम�[...]",
                "सानहাবदों आं मोथे मोथो",
            ]

    elif language == 'gu':
        test_sentences = [
                "ઓગણીસો છત્રીસ માં, પ્રથમવાર, એક્રેલીક સેફટી ગ્લાસનું, ઉત્પાદન, શરુ થ[...]",
                "વ્યાયામ પછી પ્રોટીન લેવાથી, સ્નાયુની જે પેશીયોને હાનિ પ્હોંચી હોય ��[...]"
            ]

    elif language == 'hi':
        test_sentences = [
                "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्[...]",
                "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार[...]"
            ]

    elif language == 'kn':
        test_sentences = [
                "ಯಾವುದು ನಿಜ ಯಾವುದು ಸುಳ್ಳು ಎನ್ನುವ ಬಗ್ಗೆ ಚಿಂತಿಸಿ.",
                "ಶಕ್ತಿ ಇದ್ದರೆನ್ನೊಡನೆ ಜಗಳಕ್ಕೆ ಬಾ",
            ]


    elif language == 'ml':
        test_sentences = [
                "ശിലായുഗകാലം മുതൽ മനുഷ്യർ ജ്യാമിതീയ രൂപങ്ങൾ ഉപയോഗിച്ചുവരുന്നു",
                "വാഹനാപകടത്തിൽ പരുക്കേറ്റ അധ്യാപിക മരിച്ചു",
            ]

    elif language == 'mni':
        test_sentences = [
                "মথং মথং, অসুম কাখিবনা.",
                "থেবনা ঙাশিংদু অমমম্তা ইল্লে.",
            ]

    elif language == 'mr':
        test_sentences = [
                "म्हणुनच महाराच बिरुद मी मानान वागवल",
                "घोडयावरून खाली उतरताना घोडेस्वार वृध्दाला म्हणाला, बाबा एवढया कडा�[...]"
            ]

    elif language == 'or':
        test_sentences = [
                "ସାମାନ୍ୟ ଗୋଟିଏ ବାଳକ, ସେ କ’ଣ ମହାଭାରତ ଯୁଦ୍ଧରେ ଲଢ଼ିବ ",
                "ଏ ଘଟଣା ଦେଖିବାକୁ ଶହ ଶହ ଲୋକ ଧାଇଁଲେ ",
            ]

    elif language == 'raj':
        test_sentences = [
                "कन्हैयालाल सेठिया इत्याद अनुपम काव्य कृतियां है, इंया ई, प्रकति काव�[...]",
                "नई बीनणियां रो घूंघटो नाक रे ऊपर ऊपर पड़यो सावे है",
            ]

    elif language == 'te':
        test_sentences = [
                "సింహం అడ్డువచ్చి, తప్పుకో శిక్ష విధించవలసింది నేను అని కోతిని అఙ్�[...]",
                "ఈ మాటలు వింటూనే గాలవుడు, కువలయాశ్వాన్ని ఎక్కి, శత్రుజిత్తువద్దకు �[...]"
            ]

    elif language == 'all':
        test_sentences = [
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின்[...]",
                "ఇక బిన్ లాడెన్ తర్వాతి అగ్ర నాయకులు అయ్‌మన్ అల్ జవహరి తదితర ముఖ్యు[...]",
                "ಕೆಲ ದಿನಗಳಿಂದ ಮಳೆ ಕಡಿಮೆಯಾದಂತೆ ತೋರಿದ್ದರೂ ಕಳೆದ ಎರಡು ದಿನಗಳಲ್ಲಿ ರಾಜ್ಯ��[...]",
                "കോമണ്‍വെല്‍ത്ത് ഗെയിംസ് വനിതാ ക്രിക്കറ്റ് സെമി ഫൈനലില്‍ ഇംഗ്ലണ്�[...]"
            ]

    else:
        raise ValueError("test_sentences are not defined")

    return test_sentences


def compute_attention_masks(model_path, config_path, meta_save_path, data_path, dataset_metafile, args, use_cuda=True):
    dataset_name = args.dataset_name
    language = args.language
    batch_size = 16
    meta_save_path = meta_save_path.format(dataset_name, language)

    C = load_config(config_path)
    ap = AudioProcessor(**C.audio)

    # load the model
    model = setup_model(C)
    model, _ = load_checkpoint(model, model_path, use_cuda, True)

    # data loader
    dataset_config = BaseDatasetConfig(
        name=dataset_name, 
        meta_file_train=dataset_metafile, 
        path=data_path, 
        language=language
    )
    samples, _ = load_tts_samples(
        dataset_config, 
        eval_split=False,
        formatter=formatter_indictts
    )

    dataset = TTSDataset(
        outputs_per_step=model.decoder.r if "r" in vars(model.decoder) else 1,
        compute_linear_spec=False,
        ap=ap,
        samples=samples,
        tokenizer=model.tokenizer,
        phoneme_cache_path=C.phoneme_cache_path,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=False,
    )

    # compute attentions
    file_paths = []
    with torch.no_grad():
        for data in tqdm(loader):
            # setup input data
            text_input = data["token_id"]
            text_lengths = data["token_id_lengths"]
            #linear_input = data[3]
            mel_input = data["mel"]
            mel_lengths = data["mel_lengths"]
            #stop_targets = data[6]
            item_idxs = data["item_idxs"]

            # dispatch data to GPU
            if use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            if C.model == 'glowtts':
                model_outputs = model.forward(text_input, text_lengths, mel_input, mel_lengths)
                #model_outputs = model.inference(text_input, text_lengths, mel_input, mel_lengths)
            elif C.model == 'fast_pitch':
                model_outputs = model.inference
