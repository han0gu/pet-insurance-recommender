from langchain_core.documents import Document

chunk = Document(
    page_content=('. 뚜렷한 위험의 증가와 관련된 제16조(계약 후 알릴 의무) 제1항에서 정한 계약 후<br>알릴 의무를 계약자, 피보험자 또는 이들의 '
 "대리인의 고의 또는 중대한 과실로 이행<br>하지 않았을때</p><footer id='115' "
 "style='font-size:14px'>- 10 -</footer><p id='0' data-category='paragraph' "
 "style='font-size:14px'>② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 "
 '계약을<br>해지할 수 없습니다.</p><br><p'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
