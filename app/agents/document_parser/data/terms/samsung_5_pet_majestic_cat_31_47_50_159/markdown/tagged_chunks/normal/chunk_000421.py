from langchain_core.documents import Document

chunk = Document(
    page_content=('는 전문의를 둔 병원을 말합니다.# 제3조 ( 「깁스(Cast)치료」 의 정의)① 이 특별약관에서 「깁스(Cast)치료」 라 함은 병원 '
 '또는 의원의 의사의 면허를 가진 자\n'
 '(이하 「의사」 라 합니다)가 치료를 직접적인 목적으로 「깁스(Cast)치료」 가 필요하다고 인정되는 경우로서 병원에서 의사의 관리하에 '
 '석고붕대 또는 섬유유리붕대\n'
 '(Fiberglass Cast)를 병변이 있는 뼈, 관절부위의 둘레 모두에 착용시켜(Circular Cast)\n'
 '감은 다음 굳어지게 하여 치료효과를 가져오는 치료법을 말합니다. 단, 부목(Splint'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000421',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
