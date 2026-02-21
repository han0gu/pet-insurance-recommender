from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 40 -</footer><h1 id='0' style='font-size:18px'>단체계약 "
 "보험기간 설정 추가특별약관</h1><h1 id='1' style='font-size:14px'>제1조(적용범위)</h1><br><p "
 "id='2' data-category='paragraph' style='font-size:14px'>이 추가특별약관은 단체계약 "
 '특별약관(이하“특별약관”이라 합니다) 제4조(보험의 목적의<br>증가 감소 또는 교체) 제2항에도 불구하고 새로이 증가되는 보험의 목적의'),
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
 'indexing': {'chunk_id': 'chunk_000339',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
