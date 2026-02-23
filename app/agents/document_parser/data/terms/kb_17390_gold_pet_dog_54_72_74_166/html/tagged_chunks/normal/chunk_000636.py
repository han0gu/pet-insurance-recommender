from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 회사가 "2대호흡계특정질환"의<br>조사나 확인을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 사<br>본 '
 "제출을 요청할 수 있습니다.</p><br><p id='165' data-category='list'></p><br><p id='166' "
 "data-category='paragraph' style='font-size:14px'>제4조(특별약관의 소멸)</p><br><p "
 "id='167' data-category='paragraph' style='font-size:14px'>\uf000 회사가"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000636',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
