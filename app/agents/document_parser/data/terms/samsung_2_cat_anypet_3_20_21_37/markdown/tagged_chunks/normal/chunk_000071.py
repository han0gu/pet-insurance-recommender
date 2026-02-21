from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자에게 사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계\n'
 '- 사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나 부실한 사항\n'
 '- 을 알렸다고 인정되는 경우에는 계약을 해지할 수 있습니다.\n'
 '- ⑤ 제3항에 의한 계약의 해지는 손해가 생긴 후에 이루어진 경우에도 회사는 그 손해를 보상하여 드\n'
 '- 리지 않습니다. 손해가 제3항 제1호 및 제2호의 사실로 생긴 것이 아님을 계약자 또는 피보험자가\n'
 '- 증명한 경우에는 보상하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
