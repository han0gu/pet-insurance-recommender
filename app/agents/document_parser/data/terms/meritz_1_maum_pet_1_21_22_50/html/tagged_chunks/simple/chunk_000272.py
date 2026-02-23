from langchain_core.documents import Document

chunk = Document(
    page_content=('환급)에 따른<br>보험료를 계약자에게 지급합니다.<br>④ 제1항 제1호에 따른 계약의 해지가 손해발생 후에 이루어진 경우에 회사는 그 '
 "손해를<br>보상하지 않으며, 계약 전 알릴 의무 위반 사실뿐만 아니라 계약 전 알릴 의무사항이</p><footer id='102' "
 "style='font-size:14px'>- 28 -</footer><p id='103' data-category='paragraph' "
 "style='font-size:14px'>중요한 사항에 해당되는 사유를「반대증거가 있는 경우 이의를 제기할 수"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000272',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
