from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 의무보험이 다수인 경우에는 제<br>10조(보험금의 분담)를 따릅니다.<br>② 제1항의 의무보험은 피보험자가 법률에 의하여 '
 "의무적으로 가입하여야 하는 보험으로<br>서 공제계약을 포함합니다.</p><br><h1 id='55' "
 "style='font-size:14px'>【공제계약】</h1><br><p id='56' data-category='paragraph' "
 "style='font-size:14px'>공제계약이란 동일한 직업 또는 사업에 종사하는 다수의 주체가 상호구제를 위하여<br>보험료에 "
 '상당하는 금전을 납입하고 그'),
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
 'indexing': {'chunk_id': 'chunk_000233',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
