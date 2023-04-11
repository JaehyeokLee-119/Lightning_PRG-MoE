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
🎨	`:art:`	코드의 구조/형태 개선	Improve structure / format of the code.<br>
⚡️	`:zap:`	성능 개선	Improve performance.<br>
🔥	`:fire:`	코드/파일 삭제	Remove code or files.<br>
🐛	`:bug:`	버그 수정	Fix a bug.<br>
🚑	`:ambulance:`	긴급 수정	Critical hotfix.<br>
✨	`:sparkles:`	새 기능	Introduce new features.<br>
📝	`:memo:`	문서 추가/수정	Add or update documentation.<br>
💄	`:lipstick:`	UI/스타일 파일 추가/수정	Add or update the UI and style files.<br>
🎉	`:tada:`	프로젝트 시작	Begin a project.<br>
✅	`:white_check_mark:`	테스트 추가/수정	Add or update tests.<br>
🔒	`:lock:`	보안 이슈 수정	Fix security issues.<br>
🔖	`:bookmark:`	릴리즈/버전 태그	Release / Version tags.<br>
💚	`:green_heart:`	CI 빌드 수정	Fix CI Build.<br>
📌	`:pushpin:`	특정 버전 의존성 고정	Pin dependencies to specific versions.<br>
👷	`:construction_worker:`	CI 빌드 시스템 추가/수정	Add or update CI build system.<br>
📈	`:chart_with_upwards_trend:`	분석, 추적 코드 추가/수정	Add or update analytics or track code.<br>
♻️	`:recycle:`	코드 리팩토링	Refactor code.<br>
➕	`:heavy_plus_sign:`	의존성 추가	Add a dependency.<br>
➖	`:heavy_minus_sign:`	의존성 제거	Remove a dependency.<br>
🔧	`:wrench:`	구성 파일 추가/삭제	Add or update configuration files.<br>
🔨	`:hammer:`	개발 스크립트 추가/수정	Add or update development scripts.<br>
🌐	`:globe_with_meridians:`	국제화/현지화	Internationalization and localization.<br>
💩	`:poop:`	똥싼 코드	Write bad code that needs to be improved.<br>
⏪	`:rewind:`	변경 내용 되돌리기	Revert changes.<br>
🔀	`:twisted_rightwards_arrows:`	브랜치 합병	Merge branches.<br>
📦	`:package:`	컴파일된 파일 추가/수정	Add or update compiled files or packages.<br>
👽	`:alien:`	외부 API 변화로 인한 수정	Update code due to external API changes.<br>
🚚	`:truck:`	리소스 이동, 이름 변경	Move or rename resources (e.g.: files paths routes).<br>
📄	`:page_facing_up:`	라이센스 추가/수정	Add or update license.<br>
💡	`:bulb:`	주석 추가/수정	Add or update comments in source code.<br>
🍻	`:beers:`	술 취해서 쓴 코드	Write code drunkenly.<br>
🗃	`:card_file_box:`	데이버베이스 관련 수정	Perform database related changes.<br>
🔊	`:loud_sound:`	로그 추가/수정	Add or update logs.<br>
🙈	`:see_no_evil:`	.gitignore 추가/수정	Add or update a .gitignore file.<br>