from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하여 중요한 사항을 고지하지 아니하 거나 부실의 고지를 한 때에는 '
 '보험자는 그 사실을 안 날로부터 1월내에, 계약을 체결한 날로부터 3년내에 한 하여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에 '
 '그 사실을 알았거나 중대한 과실로 인하여 알지 못 한 때에는 그러하지 아니하다. < 「상법」 제651조의2(서면에 의한 질문의 효력)> '
 '보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.\n'
 '제13조(계약 후 알릴 의무)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
