from langchain_core.documents import Document

chunk = Document(
    page_content=('- ÷ 0.5) = 6천만원\n'
 '\uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항\n'
 '각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요\n'
 '율이 변경전 요율보다 높을 때에는 회사는 그 변경사실을\n'
 '안 날로부터 1개월 이내에 계약자 또는 피보험자에게 제4항\n'
 '에 의해 보장됨을 통보하고 이에 따라 보험금을 지급합니\n'
 '다.60【중대한 과실】# 주의의무의 위반이 현저한 과실, 즉 현저한 부주의, 태\n'
 '만의 경우로서 조금만 주의를 하였다면 충분히 피해의\n'
 '발생을 막을 수 있었음에도 그 주의조차 태만히 한 높은'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
