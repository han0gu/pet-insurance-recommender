from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한 회사가 "특정정신질환"의 조사나 확인을 위하여 필요하다고 인</p><br><p id=\'10\' '
 "data-category='paragraph' style='font-size:14px'>정하는 경우 검사 결과, 진료기록부의 사본제출을 "
 "요청할 수 있습니다.</p><p id='11' data-category='paragraph' "
 "style='font-size:14px'>제4조(특별약관의 소멸)<br>피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "
 '"보험료 및<br>해약환급금 산출방법서"에서 정하는 바에 따라'),
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
 'indexing': {'chunk_id': 'chunk_000707',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
