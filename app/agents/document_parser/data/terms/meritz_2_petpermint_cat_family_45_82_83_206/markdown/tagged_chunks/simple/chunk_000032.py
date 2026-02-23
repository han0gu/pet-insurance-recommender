from langchain_core.documents import Document

chunk = Document(
    page_content=('하지 못하는 점을 고려하여, 시중의 지표금리 등에 연동\n'
 '하여 일정기간마다 변동되는 이율을 말합니다.# 【최저보증이율】회사의 운용자산이익률 및 외부지표금리가 하락하더라도\n'
 '회사에서 지급을 보증하는 최저한도의 적용이율입니다.\n'
 '예를 들어, 계약자적립액이 [보장]공시이율에 따라 적립\n'
 '되며 [보장]공시이율이 0.1%인 경우, 계약자적립액은\n'
 '[보장]공시이율(0.1%)이 아닌 최저보증이율(0.3%)로 적\n'
 '립됩니다.# 【운용자산이익률】직전 1년간의 운용자산에 대한 투자영업수익과 투자영'),
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
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
