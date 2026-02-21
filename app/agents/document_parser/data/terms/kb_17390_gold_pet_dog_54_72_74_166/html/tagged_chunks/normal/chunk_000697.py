from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제<br>도<br>또한 회사가 "환경성질환"의 조사나 확인을 위하여 필요하다고 인정하는 경우 검<br>성<br>사결과, 진료기록부의 '
 "사본제출을 요청할 수 있습니다.<br>특<br>약</p><br><p id='258' data-category='paragraph' "
 'style=\'font-size:16px\'>함은 "환경성질환 분류표"【별표15】</p><p id=\'259\' '
 "data-category='paragraph' style='font-size:18px'>- 93 -</p><br><p id='260'"),
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
 'indexing': {'chunk_id': 'chunk_000697',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
