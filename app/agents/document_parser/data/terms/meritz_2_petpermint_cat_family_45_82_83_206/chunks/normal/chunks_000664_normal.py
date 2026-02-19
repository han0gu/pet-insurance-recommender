from langchain_core.documents import Document

chunk = Document(
    page_content=('장해의 분류 | 지급률\n'
 '1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40\n'
 '2) 척추(등뼈)에 뚜렷한 운동장해를 남긴 때 | 30\n'
 '3) 척추(등뼈)에 약간의 운동장해를 남긴 때 | 10\n'
 '4) 척추(등뼈)에 심한 기형을 남긴 때 | 50\n'
 '5) 척추(등뼈)에 뚜렷한 기형을 남긴 때 | 30\n'
 '6) 척추(등뼈)에 약간의 기형을 남긴 때 | 15\n'
 '7) 추간판탈출증으로 인한 심한 신경 장해 | 20\n'
 '8) 추간판탈출증으로 인한 뚜렷한 신경 장해 | 15\n'
 '9) 추간판탈출증으로 인한 약간의 신경 장해 | 10\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 185},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000664',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
