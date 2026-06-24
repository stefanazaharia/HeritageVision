================================================================================
  GHID DE TRANSFER SI CONFIGURARE - PROIECT LICENTA
  Aplicatie AI pentru restaurarea, clasificarea si recolorarea fotografiilor vechi
================================================================================

--------------------------------------------------------------------------------
1. CERINTE DE SISTEM
--------------------------------------------------------------------------------

  - Sistem de operare : Windows 10 / 11
  - Python             : 3.11.x  (recomandat 3.11.9)
                         Download: https://www.python.org/downloads/release/python-3119/
                         ATENTIE: La instalare, bifeaza "Add Python to PATH"
  - Spatiu pe disk     : ~10 GB (venv + modele)
  - RAM recomandat     : 8 GB minim, 16 GB recomandat
  - GPU (optional)     : NVIDIA cu CUDA (pentru procesare mai rapida)
                         Fara GPU, aplicatia ruleaza pe CPU (mai lent)

--------------------------------------------------------------------------------
2. STRUCTURA PROIECTULUI
--------------------------------------------------------------------------------

  d:\LICENTA\
  |
  |-- aplicatie\                  <- Aplicatia Streamlit (interfata web)
  |   |-- app_V5_0.py             <- Fisierul principal de pornire (versiunea finala)
  |   |-- pagini\                 <- Modulele paginilor
  |   |   |-- home.py
  |   |   |-- clasificare.py      <- Detectare defecte (CNN)
  |   |   |-- restaurare4.py      <- Restaurare fotografii (versiunea finala)
  |   |   |-- recolorare.py       <- Recolorare (DDColor)
  |   |   |-- despre.py
  |   |   |-- __init__.py
  |   |-- gfpgan\weights\         <- Ponderi pentru restaurare fete
  |   |   |-- detection_Resnet50_Final.pth
  |   |   |-- parsing_parsenet.pth
  |   |-- venv311\                <- Mediul virtual Python (creat local)
  |   |-- start.bat               <- Script pornire rapida
  |   |-- pyrightconfig.json
  |
  |-- models\                     <- Modele ML antrenate
  |   |-- clasificator_versiuni\
  |   |   |-- model_clasificator_v1.pth
  |   |   |-- model_clasificator_v2.pth
  |   |   |-- model_clasificator_versiunea2_0.pth
  |   |-- restaurator_versiuni\
  |       |-- generator_restaurare_ultim.pth
  |
  |-- scripts\                    <- Scripturi de preprocesare si antrenare
  |   |-- standardizare\
  |   |-- google_colab_notebooks\
  |       |-- clasificator\
  |       |-- etichetare_CLIP\
  |       |-- restaurare\
  |
  |-- readme.txt                  <- Acest fisier
  |-- site-uri_utilizate_documentatie.txt

--------------------------------------------------------------------------------
3. PAS CU PAS - CONFIGURARE PE CALCULATOR NOU
--------------------------------------------------------------------------------

  PASUL 1: Copiaza intregul folder d:\LICENTA pe noul calculator
           (pastreaza aceeasi cale D:\LICENTA daca e posibil,
            altfel va trebui sa modifici caile din start.bat si readme)

  PASUL 2: Instaleaza Python 3.11.9
           - Download: https://www.python.org/downloads/release/python-3119/
           - La instalare bifeaza "Add Python to PATH"
           - Verifica in terminal: python --version  (trebuie sa arate 3.11.x)

  PASUL 3: Creeaza un mediu virtual NOU in folderul aplicatie\
           Deschide PowerShell si ruleaza:

             cd D:\LICENTA\aplicatie
             python -m venv venv311

  PASUL 4: Activeaza mediul virtual:

             D:\LICENTA\aplicatie\venv311\Scripts\Activate.ps1

           Daca primesti eroare de ExecutionPolicy, ruleaza mai intai:
             Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

  PASUL 5: Instaleaza toate librariile (vezi sectiunea 4 de mai jos)

  PASUL 6: Porneste aplicatia:

             cd D:\LICENTA\aplicatie
             streamlit run app_V5_0.py

           SAU dublu-click pe start.bat

  PASUL 7: Deschide browserul la adresa afisata in terminal (de obicei http://localhost:8501)

--------------------------------------------------------------------------------
4. INSTALARE LIBRARII - COMENZI PIP
--------------------------------------------------------------------------------

  Dupa activarea venv-ului (pasul 4), ruleaza urmatoarele comenzi in ordine:

  -- Librarii principale --
  pip install streamlit==1.38.0
  pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu

  [NOTA: Daca ai GPU NVIDIA cu CUDA 12.1, inlocuieste linia de torch cu:]
  pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

  pip install opencv-python==4.11.0.86
  pip install Pillow==12.2.0
  pip install numpy==1.26.4
  pip install pandas==2.3.3
  pip install scipy==1.17.1
  pip install scikit-image==0.26.0

  -- Modele AI --
  pip install basicsr==1.4.2
  pip install facexlib==0.3.0
  pip install gfpgan==1.3.8
  pip install realesrgan==0.3.0
  pip install simple-lama-inpainting==0.1.2
  pip install timm==1.0.27
  pip install modelscope==1.37.0

  -- Hugging Face si datasets --
  pip install huggingface_hub==1.15.0
  pip install datasets==4.8.5
  pip install safetensors==0.7.0

  -- Utilitare --
  pip install tqdm==4.67.3
  pip install PyYAML==6.0.3
  pip install requests==2.33.1
  pip install dlib==20.0.1
  pip install fire==0.5.0
  pip install GitPython==3.1.48
  pip install streamlit-drawable-canvas==0.9.3
  pip install einops==0.8.2
  pip install numba==0.65.1
  pip install lmdb==2.2.0
  pip install filterpy==1.4.5
  pip install matplotlib==3.10.9
  pip install ImageIO==2.37.3
  pip install addict==2.4.0
  pip install yapf==0.43.0

  -- SAU, alternativ, instaleaza tot dintr-un fisier requirements (daca exista) --
  pip install -r requirements.txt

--------------------------------------------------------------------------------
5. LISTA COMPLETA LIBRARII (pip freeze)
--------------------------------------------------------------------------------

  absl-py==2.4.0
  addict==2.4.0
  aiohappyeyeballs==2.6.1
  aiohttp==3.13.5
  aiosignal==1.4.0
  altair==5.5.0
  anyio==4.13.0
  attrs==26.1.0
  basicsr==1.4.2
  blinker==1.9.0
  cachetools==5.5.2
  certifi==2026.4.22
  charset-normalizer==3.4.7
  click==8.3.3
  colorama==0.4.6
  contourpy==1.3.3
  cycler==0.12.1
  datasets==4.8.5
  dill==0.4.1
  dlib==20.0.1
  einops==0.8.2
  facexlib==0.3.0
  filelock==3.29.0
  filterpy==1.4.5
  fire==0.5.0
  fonttools==4.62.1
  frozenlist==1.8.0
  fsspec==2026.2.0
  future==1.0.0
  gfpgan==1.3.8
  gitdb==4.0.12
  GitPython==3.1.48
  grpcio==1.80.0
  huggingface_hub==1.15.0
  ImageIO==2.37.3
  Jinja2==3.1.6
  jsonschema==4.26.0
  kiwisolver==1.5.0
  lazy-loader==0.5
  llvmlite==0.47.0
  lmdb==2.2.0
  Markdown==3.10.2
  markdown-it-py==4.0.0
  MarkupSafe==3.0.3
  matplotlib==3.10.9
  modelscope==1.37.0
  mpmath==1.3.0
  multidict==6.7.1
  multiprocess==0.70.19
  networkx==3.6.1
  numba==0.65.1
  numpy==1.26.4
  opencv-python==4.11.0.86
  packaging==24.2
  pandas==2.3.3
  pillow==12.2.0
  platformdirs==4.9.6
  protobuf==5.29.6
  pyarrow==24.0.0
  pydeck==0.9.2
  Pygments==2.20.0
  pyparsing==3.3.2
  python-dateutil==2.9.0.post0
  pytz==2026.1.post1
  PyYAML==6.0.3
  realesrgan==0.3.0
  requests==2.33.1
  rich==13.9.4
  safetensors==0.7.0
  scikit-image==0.26.0
  scipy==1.17.1
  simple-lama-inpainting==0.1.2
  simplejson==4.1.1
  six==1.17.0
  smmap==5.0.3
  sortedcontainers==2.4.0
  streamlit==1.38.0
  streamlit-drawable-canvas==0.9.3
  sympy==1.13.1
  tenacity==8.5.0
  tensorboard-data-server==0.7.2
  termcolor==3.3.0
  tifffile==2026.3.3
  timm==1.0.27
  toml==0.10.2
  torch==2.5.0
  torchvision==0.20.0
  tornado==6.5.5
  tqdm==4.67.3
  typer==0.25.1
  typing_extensions==4.15.0
  tzdata==2026.2
  urllib3==2.6.3
  watchdog==4.0.2
  Werkzeug==3.1.8
  xxhash==3.7.0
  yapf==0.43.0
  yarl==1.23.0

--------------------------------------------------------------------------------
6. MODELE CARE SE DESCARCA AUTOMAT LA PRIMA RULARE
--------------------------------------------------------------------------------

  Urmatoarele modele NU sunt incluse in proiect si se descarca automat
  din internet la prima utilizare a functiei respective:

  - RealESRGAN_x2plus.pth     (~67 MB)  - Super-rezolutie 2x
    (descarcat de realesrgan din GitHub la prima rulare a restaurarii)

  - GFPGANv1.3.pth            (~348 MB) - Restaurare fete
    (descarcat de gfpgan la prima rulare a restaurarii cu detectare fete)

  - DDColor model              (~800 MB) - Recolorare fotografii
    (descarcat de modelscope la prima rulare a recolorarii)

  ATENTIE: Este nevoie de conexiune la internet la PRIMA rulare.
           Modelele se salveaza in cache local dupa prima descarcare.

--------------------------------------------------------------------------------
7. MODELE INCLUSE IN PROIECT (trebuie copiate manual)
--------------------------------------------------------------------------------

  Urmatoarele fisiere .pth TREBUIE sa existe pe noul calculator:

  D:\LICENTA\models\clasificator_versiuni\
    - model_clasificator_v1.pth
    - model_clasificator_v2.pth
    - model_clasificator_versiunea2_0.pth        <- MODELUL FOLOSIT IN APLICATIE

  D:\LICENTA\models\restaurator_versiuni\
    - generator_restaurare_ultim.pth             <- MODELUL FOLOSIT IN APLICATIE

  D:\LICENTA\aplicatie\gfpgan\weights\
    - detection_Resnet50_Final.pth
    - parsing_parsenet.pth

  Daca aceste fisiere lipsesc, aplicatia va da eroare la pornire.

--------------------------------------------------------------------------------
8. COMENZI UTILE
--------------------------------------------------------------------------------

  Activare mediu virtual (PowerShell):
    D:\LICENTA\aplicatie\venv311\Scripts\Activate.ps1

  Activare mediu virtual (CMD):
    D:\LICENTA\aplicatie\venv311\Scripts\activate.bat

  Pornire aplicatie:
    streamlit run D:\LICENTA\aplicatie\app_V5_0.py

  Verificare versiune Python:
    python --version

  Verificare librarii instalate:
    pip list

  Generare requirements.txt (pentru backup):
    pip freeze > requirements.txt

  Dezactivare mediu virtual:
    deactivate

--------------------------------------------------------------------------------
9. REZOLVARE PROBLEME COMUNE
--------------------------------------------------------------------------------

  PROBLEMA: "ExecutionPolicy" error la activarea venv in PowerShell
  SOLUTIE:  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

  PROBLEMA: "ModuleNotFoundError: No module named 'X'"
  SOLUTIE:  Asigura-te ca venv-ul este activ, apoi: pip install X

  PROBLEMA: Aplicatia porneste dar recolorarea nu functioneaza
  SOLUTIE:  Prima rulare necesita internet pentru descarcarea modelului DDColor

  PROBLEMA: Eroare CUDA / GPU
  SOLUTIE:  Aplicatia functioneaza si pe CPU. Instaleaza versiunea CPU de torch:
            pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cpu

  PROBLEMA: dlib nu se instaleaza
  SOLUTIE:  Instaleaza mai intai Visual C++ Build Tools:
            https://visualstudio.microsoft.com/visual-cpp-build-tools/
            Sau descarca wheel-ul precompilat pentru Python 3.11:
            pip install dlib==20.0.1 (poate necesita compilatoare C++)

  PROBLEMA: Modelele .pth nu sunt gasite
  SOLUTIE:  Verifica ca fisierele din sectiunea 7 exista la caile specificate.
            Verifica caile hardcodate din pagini/clasificare.py si pagini/restaurare4.py

--------------------------------------------------------------------------------
10. VERSIUNI EXACTE FOLOSITE
--------------------------------------------------------------------------------

  Python           : 3.11.9
  Streamlit        : 1.38.0
  PyTorch          : 2.5.0
  TorchVision      : 0.20.0
  OpenCV           : 4.11.0.86
  Pillow           : 12.2.0
  NumPy            : 1.26.4
  Pandas           : 2.3.3
  SciPy            : 1.17.1
  Scikit-image     : 0.26.0
  GFPGAN           : 1.3.8
  BasicSR          : 1.4.2
  FaceXLib         : 0.3.0
  RealESRGAN       : 0.3.0
  SimpleLama       : 0.1.2
  Timm             : 1.0.27
  ModelScope       : 1.37.0
  HuggingFace Hub  : 1.15.0
  Datasets         : 4.8.5
  Safetensors      : 0.7.0
  DLib             : 20.0.1
  Tqdm             : 4.67.3
  PyYAML           : 6.0.3
  Requests         : 2.33.1
  Einops           : 0.8.2
  Numba            : 0.65.1
  Matplotlib       : 3.10.9
  ImageIO          : 2.37.3

================================================================================
