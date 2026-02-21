from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 제4조(특별약관의 소<br>멸) 제2항에 따라 이 특별약관의 계약자적립액 등을 지급한 경우에는, 이미 지급<br>된 '
 "계약자적립액 등을 차감하고 그 차액을 지급합니다.</p><br><p id='160' "
 "data-category='list'></p><br><h1 id='161' "
 "style='font-size:14px'>제3조(2대호흡계특정질환의 정의 및 진단확정)</h1><br><h1 id='162' "
 'style=\'font-size:14px\'>\uf000 이 특별약관에 있어서 "2대호흡계특정질환"이라</h1><br><p '
 "id='163'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000633',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
