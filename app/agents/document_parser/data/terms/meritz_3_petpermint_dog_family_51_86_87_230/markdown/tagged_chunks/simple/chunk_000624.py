from langchain_core.documents import Document

chunk = Document(
    page_content=('주한다.# 6. 척추(등뼈)의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40 |\n'
 '| 2) 척추(등뼈)에 뚜렷한 운동장해를 남긴 때 | 30 |\n'
 '| 3) 척추(등뼈)에 약간의 운동장해를 남긴 때 | 10 |\n'
 '| 4) 척추(등뼈)에 심한 기형을 남긴 때 | 50 |\n'
 '| 5) 척추(등뼈)에 뚜렷한 기형을 남긴 때 | 30 |\n'
 '| 6) 척추(등뼈)에 약간의 기형을 남긴 때 | 15 |\n'
 '| 7) 추간판탈출증으로 인한 심한 신경 장해 | 20 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000624',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
