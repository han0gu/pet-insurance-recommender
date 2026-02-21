from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 회사가 "6대호흡계특정질환"의<br>조사나 확인을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 '
 "사</p><br><h1 id='186' style='font-size:16px'>본 제출을 요청할 수 있습니다.</h1><p "
 "id='187' data-category='list' style='font-size:14px'>제4조(특별약관의 소멸) "
 '상<br>\uf000 회사가 제1조(보험금의 지급사유)에서 정한 6대호흡계특정질환진단비를 지급한 해<br>경우에는 그 지급사유가 발생한 '
 '때부터 이 특별약관 계약은 소멸되며 이'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000648',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
