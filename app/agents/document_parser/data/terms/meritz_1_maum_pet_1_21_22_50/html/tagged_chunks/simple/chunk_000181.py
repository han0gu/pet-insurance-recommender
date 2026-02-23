from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢금<br>융소비자보호에 관한 법률｣ 제42조에서 정하는 일정 금액 이내인 '
 '분쟁사건에 대하여<br>조정절차가 개시된 경우에는 관계 법령이 정하는 경우를 제외하고는 소를 제기하지 않<br>습니다.</p><h1 '
 "id='99' style='font-size:14px'>제35조(관할법원)</h1><br><p id='100' "
 "data-category='paragraph' style='font-size:14px'>이 계약에 관한 소송 및 민사조정은 계약자의 "
 '주소지를 관할하는 법원으로'),
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
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
