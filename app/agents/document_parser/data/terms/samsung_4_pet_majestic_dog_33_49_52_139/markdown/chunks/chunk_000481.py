from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한다.\n'
 '- 2. "동물"이란 소, 말, 돼지, 양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생동물(水生動物), 그 밖\n'
 '- 에 대통령령으로 정하는 동물을 말한다.\n'
 '- 3. "동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물\n'
 '- 의 질병을 예방하는 업(業)을 말한다.\n'
 '- 3의2. "동물보건사"란 동물병원 내에서 수의사의 지도 아래 동물의 간호 또는 진료 보조 업무\n'
 '- 에 종사하는 사람으로서 농림축산식품부장관의 자격인정을 받은 사람을 말한다.'),
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
