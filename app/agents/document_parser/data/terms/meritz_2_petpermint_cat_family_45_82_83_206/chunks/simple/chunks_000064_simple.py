from langchain_core.documents import Document

chunk = Document(
    page_content=('【중대한 과실】\n'
 '주의의무의 위반이 현저한 과실, 즉 현저한 부주의, 태 만의 경우로서 조금만 주의를 하였다면 충분히 피해의 발생을 막을 수 있었음에도 그 '
 '주의조차 태만히 한 높은 강도의 주의의무위반\n'
 '제17조(알릴 의무 위반의 효과)\n'
 '\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생 여부에 관계없이 이 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 61},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
