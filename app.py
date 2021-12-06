from flask import Flask, render_template, request
import os
import json 
app = Flask(__name__)

@app.route('/prdictions', methods=['POST','GET'])
def predict():
    result = os.system("""
    python fairseq/scripts/spm_encode.py \
    --model model/sentence.bpe.model \
    --inputs Data/test.good Data/test.bad \
    --outputs model/spm/test.bpe.good model/spm/test.bpe.bad
    """)

    preprocess = os.system("""fairseq-preprocess \
                                --source-lang "bad" \
                                --target-lang "good" \
                                --testpref model/spm/test.bpe \
                                --destdir model/preprocess \
                                --srcdict model/preprocess/dict.si_LK.txt \
                                --tgtdict model/preprocess/dict.si_LK.txt \
                                --thresholdtgt 0 \
                                --thresholdsrc 0 \
                                --workers 70""")
                                
    print("SPM Encode: ", result)
    print("Preprocess: ", preprocess)
    return json.dumps({
        "spm":result,
        "preprocess":preprocess
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)