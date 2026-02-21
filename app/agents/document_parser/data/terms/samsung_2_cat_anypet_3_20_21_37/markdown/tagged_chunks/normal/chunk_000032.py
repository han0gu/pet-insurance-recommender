from langchain_core.documents import Document

chunk = Document(
    page_content=('대하여 알고 있는 사실을 반드시 사실대로 알려야 합니다.# 【관련법규】く 「상법」 제651조(고지의무위반으로 인한 계약해지)>\n'
 '보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하여 중요한 사항을 고지하지 아니하\n'
 '거나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날로부터 1월내에, 계약을 체결한 날로부터 3년내에 한\n'
 '하여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에 그 사실을 알았거나 중대한 과실로 인하여 알지 못\n'
 '한 때에는 그러하지 아니하다.< 「상법」 제651조의2(서면에 의한 질문의 효력)>'),
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
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
