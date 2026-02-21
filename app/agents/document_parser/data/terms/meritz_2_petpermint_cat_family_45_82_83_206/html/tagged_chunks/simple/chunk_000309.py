from langchain_core.documents import Document

chunk = Document(
    page_content=('있을 경우 각 계<br>약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 지<br>급보험금의 합계액이 피보험자가 부담한 비용금액을 '
 "초과할<br>때에는 아래에 따라 보험금을 지급합니다.</p><br><p id='42' data-category='paragraph' "
 "style='font-size:20px'>피보험자가 이 계약의 지급보험금<br>부담한 총 × 다른 계약이 없는 것으로 하여 각각 "
 "계산한<br>비용금액 지급보험금의 합계액</p><br><p id='43' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000309',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
