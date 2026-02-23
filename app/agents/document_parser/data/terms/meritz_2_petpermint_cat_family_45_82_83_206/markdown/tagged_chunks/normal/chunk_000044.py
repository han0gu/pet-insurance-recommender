from langchain_core.documents import Document

chunk = Document(
    page_content=('여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에\n'
 '그 사실을 알았거나 중대한 과실로 인하여 알지 못한 때\n'
 '에는 그러하지 아니하다.# 【상법 제651조의2(서면에 의한 질문의 효력)】보험자가 서면으로 질문한 사항은 중요한 사항으로 추정\n'
 '한다.# 【사례】계약 청약을 하면서 보험설계사에게 고혈압이 있다고만\n'
 '얘기하였을 뿐, 청약서의 계약 전 알릴 사항에 아무런\n'
 '기재도 하지 않았을 경우에는 보험설계사에게만 고혈압\n'
 '병력을 얘기하였다고 하더라도 회사는 계약 전 알릴 의\n'
 '무 위반을 이유로 계약을 해지하고 보험금을 지급하지'),
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
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
