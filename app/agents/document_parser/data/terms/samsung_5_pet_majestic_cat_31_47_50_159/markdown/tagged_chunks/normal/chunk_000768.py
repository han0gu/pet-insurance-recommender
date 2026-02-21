from langchain_core.documents import Document

chunk = Document(
    page_content=('| 4) 척추(등뼈)에 심한 기형을 남긴 때 | 50 |\n'
 '| 5) 척추(등뼈)에 뚜렷한 기형을 남긴 때 | 30 |\n'
 '| 6) 척추(등뼈)에 약간의 기형을 남긴 때 | 15 |\n'
 '| 7) 추간판탈출증으로 인한 심한 신경 장해 | 20 |\n'
 '| 8) 추간판탈출증으로 인한 뚜렷한 신경 장해 | 15 |\n'
 '| 9) 추간판탈출증으로 인한 약간의 신경 장해 | 10 |\n'
 '# 나. 장해판정기준1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일한 부위로 한다. 제2천추\n'
 '이하의 천골 및 미골은 체간골의 장해로 평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000768',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
