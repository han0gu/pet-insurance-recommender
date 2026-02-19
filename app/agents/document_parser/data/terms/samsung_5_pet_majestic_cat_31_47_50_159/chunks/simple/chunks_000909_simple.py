from langchain_core.documents import Document

chunk = Document(
    page_content=('9) 추간판탈출증으로 인한 약간의 신경 장해 | 10\n'
 '나. 장해판정기준\n'
 '1) 척추(등뼈)는 경추에서 흉추, 요추, 제1천추까지를 동일한 부위로 한다. 제2천추 이하의 천골 및 미골은 체간골의 장해로 평가한다. '
 '2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 말하며, 횡돌기 및 극돌기는 제 외한다. 이하 이 신체부위에서 같다)의 압박률 또는 '
 '척추체(척추뼈 몸통)의 만 곡 정도에 따라 평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000909',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
