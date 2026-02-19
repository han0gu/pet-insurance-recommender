from langchain_core.documents import Document

chunk = Document(
    page_content=('7cm | 5cm이상 성형수술에 해당 | 5cm이상 성형수술비 = 50만원\n'
 '② 제1항에서 정한 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다. ③ 제1항에서 길이측정이 불가한 피부이식수술 등의 경우 '
 '수술cm는 최장직경으로 합니 다. ④ 제1항의 안면부 상해흉터복원(성형) 수술비는 매사고마다 지급합니다. 다만, 동일부위 에 대한 '
 '성형수술을 2회 이상 받은 경우에는 최초로 받은 수술에 대해서만 지급합니 다.\n'
 '제 2조 (수술의 정의와 장소)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 83},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000440',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
