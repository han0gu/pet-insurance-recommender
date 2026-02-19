from langchain_core.documents import Document

chunk = Document(
    page_content=('【핵연료물질】\n'
 '사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】\n'
 '원자핵분열 생성물을 포함합니다.\n'
 '⑥ 피보험자의 피고용인이 피보험자의 업무에 종사중에 입은 신체의 장해로 인한 배상책임 ⑦ 피보험자와 타인간에 손해배상에 관한 약정이 있는 '
 '경 우 그 약정에 따라 가중된 배상책임 ⑧ 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대하여 정당한 '
 '권리를 가진 사람에게 부담하는 배상책임 ⑨ 피보험자의 심신상실로 인한 배상책임'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 187},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000639',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
