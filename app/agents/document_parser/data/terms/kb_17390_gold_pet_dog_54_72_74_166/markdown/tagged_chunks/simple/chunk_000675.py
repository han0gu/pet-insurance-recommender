from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6. 국가동물 미등록한 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를\n'
 '- 회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금\n'
 '- 을 지급합니다.\n'
 '# 제8조(보험금의# 지급절차)- \uf000 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고\n'
 '- 휴대전화 문자메시지 또는 전자우편 등으로도 송부하며, 접수 후 지체없이 지급\n'
 '- 할 보험금을 결정하고 지급할 보험금이 결정되면 7일 이내에 이를 지급합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000675',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
