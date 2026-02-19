from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자, 피보험자 또는 보험수익자가 보험금을 지급받 을 목적으로 고의로 보험금 지급사유를 발생시킨 경 우 ② 계약자, 피보험자 또는 '
 '보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 '
 '보험금 지급사유가 발생한 경우에는 이에 대한 보험금은 지급합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000149',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
