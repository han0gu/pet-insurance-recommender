from langchain_core.documents import Document

chunk = Document(
    page_content=('1) "손바닥 크기" 라 함은 해당 환자의 손가락을 제외한 손바닥의 크기를 말하며,\n'
 '12세 이상의 성인에서는 8× 10cm(1/2 크기는 40cm2, 1/4 크기는 20cm2), 6~11 세의 경우는 6× 8cm(1/2 '
 '크기는 24cm2, 1/4 크기는 12cm2), 6세 미만의 경우는 4× 6cm(1/2 크기는 12cm2, 1/4 크기는 6cm2)로 '
 '간주한다.\n'
 '6. 척추(등뼈)의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000907',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
