from langchain_core.documents import Document

chunk = Document(
    page_content=('【 신의성실의 원칙 】\n'
 '권리의 행사와 의무의 이행은 신의와 성실을 가지고 행동 하여 상대방의 신뢰와 기대를 배반하여서는 안된다는 원칙 (「민법」제2조 제1항)\n'
 '【 민법 제2조(신의성실) 제1항 】\n'
 '① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하 여야 한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 79},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
