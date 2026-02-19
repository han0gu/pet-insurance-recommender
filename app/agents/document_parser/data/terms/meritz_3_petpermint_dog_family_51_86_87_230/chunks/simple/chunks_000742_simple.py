from langchain_core.documents import Document

chunk = Document(
    page_content=('한 부위로 한다. 제2천추 이하의 천골 및 미골은 체간골 의 장해로 평가한다.\n'
 '2) 척추(등뼈)의 기형장해는 척추체(척추뼈 몸통을 말하며, 횡돌기 및 극돌기는 제외한다. 이하 이 신체부위에서 같 다)의 압박률 또는 '
 '척추체(척추뼈 몸통)의 만곡 정도에 따라 평가한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 211},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000742',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
