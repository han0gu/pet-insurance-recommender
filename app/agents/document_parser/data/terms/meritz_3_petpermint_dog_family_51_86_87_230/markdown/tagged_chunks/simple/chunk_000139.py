from langchain_core.documents import Document

chunk = Document(
    page_content=('정도로 현저하게 공정성을 잃은 것을 말합니다.# 제46조(개인정보보호)\uf000 회사는 이 계약과 관련된 개인정보를 이 계약의 체결,\n'
 '유지, 보험금 지급 등을 위하여「개인정보 보호법」,「신용\n'
 '정보의 이용 및 보호에 관한 법률」등 관계 법령에 정한 경\n'
 '우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없85이 수집, 이용, 조회 또는 제공하지 않습니다. 다만, 회사\n'
 '는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위 관\n'
 '계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
