from langchain_core.documents import Document

chunk = Document(
    page_content=('술기법으로 생체에 절단, 절제 등의 조작을 가하는 것을 포함합니다. 또한 레이저\n'
 '(Laser)를 이용하여 생체에 절단, 절제 등의 조작을 가하는 것도 포함됩니다.<용어풀이># [신의료기술평가위원회]의료법 '
 '제54조(신의료기 술평가위원회의 설치 등)에 의거 설치된 위원회로서 신의료기술에 관한\n'
 '최고의 심의기구를 말합니다.# ③ 제1항의 수술에서 아래에 정한 사항은 제외합니다.1. 흡인(吸引, 주사기 등으로 빨아들이는 것)\n'
 '2. 천자(穿刺, 바늘 또는 관을 꽂아 체액·조직을 뽑아내거나 약물을 주입하는 것) 등의\n'
 '조치'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
