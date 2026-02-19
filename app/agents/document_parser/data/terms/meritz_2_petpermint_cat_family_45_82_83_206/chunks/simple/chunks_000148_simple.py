from langchain_core.documents import Document

chunk = Document(
    page_content=('【위법계약】\n'
 '금융소비자보호에 관한 법률 제47조에서 정한 적합성원 칙, 적정성원칙, 설명의무, 불공정영업행위 금지 또는 부당권유행위 금지를 위반한 '
 '계약을 말합니다.\n'
 '제33조(중대사유로 인한 해지)\n'
 '\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1 개월 이내에 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 165,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
