from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 계약의 평균 공시이율은 2.75%입니다.</td></tr><tr><td>계약자적 립액</td><td>장래의 해약환급금 등을 '
 '지급하기 위하여 계약 자가 납입한 보험료 중 일정액을 기준으로 보 험료 및 해약환급금 산출방법서에서 정한 방법 에 따라 계산한 금액을 '
 "말합니다.</td></tr></tbody></table><footer id='86' "
 "style='font-size:14px'>86</footer><table id='0'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000280',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
