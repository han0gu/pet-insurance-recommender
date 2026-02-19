from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 위하여「개인정보 보호법」,「신용 정보의 이용 '
 '및 보호에 관한 법률」등 관계 법령에 정한 경 우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없 이 수집, 이용, 조회 또는 '
 '제공하지 않습니다. 다만, 회사 는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위 관 계 법령에 따라 계약자 및 피보험자의 동의를 '
 '받아 다른 보 험회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다. \uf000 회사는 계약과 관련된 개인정보를 안전하게 '
 '관리하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 81},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
