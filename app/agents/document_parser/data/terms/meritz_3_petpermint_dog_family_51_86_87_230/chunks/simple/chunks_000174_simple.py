from langchain_core.documents import Document

chunk = Document(
    page_content=('【 현저하게 공정을 잃은 합의 】\n'
 '사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정성을 잃은 것을 말합니다.\n'
 '제46조(개인정보보호)\n'
 '\uf000 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 위하여「개인정보 보호법」,「신용 정보의 이용 '
 '및 보호에 관한 법률」등 관계 법령에 정한 경 우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 85},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000174',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
