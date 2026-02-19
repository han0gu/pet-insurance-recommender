from langchain_core.documents import Document

chunk = Document(
    page_content=('【[보장]공시이율】\n'
 '전통적인 보험상품에 적용되는 이율이 장기･고정금리이 기 때문에 시중금리가 급격하게 변동할 경우 이에 대응 하지 못하는 점을 고려하여, '
 '시중의 지표금리 등에 연동 하여 일정기간마다 변동되는 이율을 말합니다.\n'
 '【최저보증이율】\n'
 '회사의 운용자산이익률 및 외부지표금리가 하락하더라도 회사에서 지급을 보증하는 최저한도의 적용이율입니다. 예를 들어, 계약자적립액이 '
 '[보장]공시이율에 따라 적립 되며 [보장]공시이율이 0.1%인 경우, 계약자적립액은 [보장]공시이율(0.1%)이 아닌 '
 '최저보증이율(0.3%)로 적 립됩니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 59},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
