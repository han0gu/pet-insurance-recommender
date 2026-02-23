from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 추간판탈출증으로 인한 심한 신경 장해 | 20 |\n'
 '| 8) 추간판탈출증으로 인한 뚜렷한 신경 장해 | 15 |\n'
 '| 9) 추간판탈출증으로 인한 약간의 신경 장해 | 10 |\n'
 '# 나. 장해판정기준1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일210한 부위로 한다. 제2천추 이하의 천골 및 미골은 '
 '체간골\n'
 '의 장해로 평가한다.2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 말하며,\n'
 '횡돌기 및 극돌기는 제외한다. 이하 이 신체부위에서 같\n'
 '다)의 압박률 또는 척추체(척추뼈 몸통)의 만곡 정도에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
