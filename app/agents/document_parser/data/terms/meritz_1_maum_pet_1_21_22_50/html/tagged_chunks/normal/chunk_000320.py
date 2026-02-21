from langchain_core.documents import Document

chunk = Document(
    page_content=('보<br>장조건 및 인수기준에 따라 가입될 수 있으며, 보험의 목적 교체시점부터 잔여 보험기<br>간(보험의 목적 교체전 계약의 보험기간 '
 "만료일)까지 보상하여 드립니다.</p><h1 id='67' style='font-size:14px'>제5조(개별계약으로의 "
 "전환)</h1><br><p id='68' data-category='list' style='font-size:14px'>① 피보험자가 "
 '퇴직 등의 사유로 인하여 피보험단체에서 탈퇴하는 경우 피보험자가 보험<br>료의 일부를 부담한 경우에 한하여 탈퇴일로부터 1개월 이내에 '
 '계약자'),
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
 'indexing': {'chunk_id': 'chunk_000320',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
