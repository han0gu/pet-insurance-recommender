from langchain_core.documents import Document

chunk = Document(
    page_content=('된 계약자적립액 등을 차감하고 그 차액을 지급합니다.- \n'
 '# 제3조(2대호흡계특정질환의 정의 및 진단확정)# \uf000 이 특별약관에 있어서 "2대호흡계특정질환"이라함은 제9차 '
 '한국표준질병․사인분류에 있어서 【별표11】(2대호흡계특정질환 분류표)에서 정한 질병을 말합니다.\n'
 '\uf000 "2대호흡계특정질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병\n'
 '원이나 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외)\n'
 '면허를 가진 자에 의하여 내려져야 합니다. 또한, 회사가 "2대호흡계특정질환"의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000387',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
