# Lightning_PRG-MoE

Non-official implementation of [Conversational Emotion-Cause Pair Extraction with Guided Mixture of Experts](https://github.com/jdjin3000/PRG-MoE)

수행할 내용: <br>
추가적인 실험을 위해 다양한 pre-trained model에 대해 generalize, additional metrics, fine-tuning methods 도입 <br>
pytorch Lightning을 활용<br> 

[PRG-MoE](https://github.com/jdjin3000/PRG-MoE) referenced

## Dependencies
- python 3.10.10<br>
- pytorch 1.13.1<br>
- pytorch-cuda 11.6<br>
- tqdm 4.64.1<br>
- numpy 1.23.5<br>
- huggingface_hub 0.12.0<br>
- cuda 11.6.1<br>
- transformers 4.26.1<br>
- scikit-learn 1.2.0<br>
- pytorch-lightning 2.0.1<br>
- peft: !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

## Dataset
The dataset used in this model is [RECCON dataset](https://github.com/declare-lab/RECCON)


### Commit Conventions (from https://treasurebear.tistory.com/70)
🎉	`:tada:`	프로젝트 시작	Begin a project.<br>
📝	`:memo:`	문서 추가/수정	Add or update documentation.<br>
🎨	`:art:`	문서의 시각적 개선	Improve structure / format of the documents.<br>
✨	`:sparkles:`	코드의 구조/형태 개선	Improve structure / format of the code.<br>
📈	`:chart_with_upwards_trend:`	시각화/분석 추가	Visualization, analyze feature.<br>
⚡️	`:zap:`	새 기능	Introduce new features.<br>
🔥	`:fire:`	코드/파일 삭제	Remove code or files.<br>
🐛	`:bug:`	버그 발견 Find a bug.<br>
🔨	`:hammer:`	버그 수정 Fix a bug.<br>
🙈	`:see_no_evil:`	.gitignore 추가/수정	Add or update a .gitignore file.<br>
💡	`:bulb:`	주석 추가/수정	Add or update comments in source code.<br>
💩	`:poop:`	똥싼 코드	Write bad code that needs to be improved.<br>
🍻	`:beers:`	술 취해서 쓴 코드	Write code drunkenly.<br>
📦	`:package:`	데이터베이스 관련 수정	Perform database related changes.<br>