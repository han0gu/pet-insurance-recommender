from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 피상속인의 형제자매 ④ 피상속인의 4촌 이내의 방계혈족\n'
 '# [직계비속]자기로부터 직계로 이어져 내려가는 혈족. 자녀, 손자, 증손 등\n'
 '[직계존속]\n'
 '조상으로부터 직계로 내려와 자기에 이르는 사이의 혈족. 부모, 조부모 등\n'
 '[방계혈족]\n'
 '자기의 형제자매와 형제자매의 직계비속, 직계존속의 형제자매 및 그 형제자매의 직계비속④ 보험수익자는 통지를 받은 날(제3항에 따라 '
 '계약자에게 통지된 경우에는 계약자가 통'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000533',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
