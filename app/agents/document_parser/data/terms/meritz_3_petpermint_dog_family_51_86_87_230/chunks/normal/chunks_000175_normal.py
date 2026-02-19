from langchain_core.documents import Document

chunk = Document(
    page_content=('이 수집, 이용, 조회 또는 제공하지 않습니다. 다만, 회사 는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위 관 계 법령에 따라 '
 '계약자 및 피보험자의 동의를 받아 다른 보 험회사 및 보험관련단체 등에 개인정보를 제공할 수 있습니다. \uf000 회사는 계약과 관련된 '
 '개인정보를 안전하게 관리하여야 합니다.\n'
 '제47조(준거법)\n'
 '이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에 서 정하지 않은 사항은 금융소비자보호에 관한 법률, 상법, 민법 등 관계 법령을 '
 '따릅니다.\n'
 '제48조(예금보험에 의한 지급보장)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 86},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
