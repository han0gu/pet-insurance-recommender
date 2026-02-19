from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제11조(손해방지의무)\n'
 '보험사고가 생긴 때에는 계약자 또는 피보험자는 손해의 방지와 경감에 힘써야 합니다. 만약, 계약자 또는 피보험자가 고의 또는 중대한 '
 '과실로 이를 게을리한 때에는 방지 또는 경감할 수 있었을 것으로 밝혀진 값을 손해액에서 뺍니다.\n'
 '제3관 계약자의 계약 전 알릴 의무 등\n'
 '제12조(계약 전 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000034',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
