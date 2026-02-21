from langchain_core.documents import Document

chunk = Document(
    page_content=('통령령으로 정하는 동물을 말한다.\n'
 '3. "동물진료업"이란 동물을 진료[동물의 사체 검안(檢案)을 포함한다. 이하 같다]하거나 동물의 질\n'
 '병을 예방하는 업(業)을 말한다.\n'
 '3의2. "동물보건사"란 동물병원 내에서 수의사의 지도 아래 동물의 간호 또는 진료 보조 업무에 종\n'
 '사하는 사람으로서 농림축산식품부장관의 자격인정을 받은 사람을 말한다.\n'
 '4. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 한 진료기관을 말한다.- \n'
 '③ 제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국'),
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
