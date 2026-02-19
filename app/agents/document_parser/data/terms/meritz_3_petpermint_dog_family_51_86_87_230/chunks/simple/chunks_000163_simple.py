from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하 여야 하며 계약자에 따라 다르게 해석하지 않습니다.\n'
 '【 신의성실의 원칙 】\n'
 '권리의 행사와 의무의 이행은 신의와 성실을 가지고 행동 하여 상대방의 신뢰와 기대를 배반하여서는 안된다는 원칙 (「민법」제2조 제1항)\n'
 '【 민법 제2조(신의성실) 제1항 】\n'
 '① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하 여야 한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 84},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
