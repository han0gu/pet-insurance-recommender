from langchain_core.documents import Document

chunk = Document(
    page_content=('별\n'
 '\uf000 "6대호흡계특정질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병\n'
 '약\n'
 '원이나 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외)\n'
 '관\n'
 '면허를 가진 자에 의하여 내려져야 합니다. 또한, 회사가 "6대호흡계특정질환"의\n'
 '조사나 확인을 위하여 필요하다고 인정하는 경우에는 검사결과, 진료기록부의 사# 본 제출을 요청할 수 있습니다.- 제4조(특별약관의 소멸) '
 '상\n'
 '- \uf000 회사가 제1조(보험금의 지급사유)에서 정한 6대호흡계특정질환진단비를 지급한 해'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000394',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
