from langchain_core.documents import Document

chunk = Document(
    page_content=('- 로 납입한 계약의 청약을 철회하는 경우에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내에\n'
 '- 해당 신용카드회사로 하여금 대금청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료를 반환한\n'
 '- 것으로 봅니다.\n'
 '- ⑤ 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으나 계약자가 그 보험금 지급사유가 발생한\n'
 '- 사실을 알지 못한 경우에는 청약철회의 효력은 발생하지 않습니다.\n'
 '- ⑥ 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
