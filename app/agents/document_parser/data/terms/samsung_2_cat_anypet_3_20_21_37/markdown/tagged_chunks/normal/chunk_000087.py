from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 관한 중요한 사항을 계약자가 이해할 수 있도록 설명하고 계약자가 이해하였음을 서명( 「전자\n'
 '- 서명법」 제2조 제2호에 따른 전자서명을 포함), 기명날인 또는 녹취 등을 통해 확인받아야 하며,\n'
 '- 설명서를 제공하여야 합니다.\n'
 '- ② 설명서, 약관, 청약서 부본 및 증권의 제공 사실에 관하여 계약자와 회사간에 다툼이 있는 경우에\n'
 '- 는 회사가 이를 증명하여야 합니다.\n'
 '- ③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료의 내용이 약관의 내용과 다른 경'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
