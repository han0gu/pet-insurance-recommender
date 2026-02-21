from langchain_core.documents import Document

chunk = Document(
    page_content=('- 리에 대한 태만\n'
 '- 10. 동물보호법 위반 등 동물학대에 기인하는 손해\n'
 '11. 사망사실을 명확하게 입증할 수 없는 실종, 행방불명 등# 제4조(보험금의 청구)- 보험수익자는 다음의 서류를 제출하고 보험금을 '
 '청구하여야 합니다.\n'
 '- 1. 청구서(회사 양식)\n'
 '- 2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 국가동물 미등록한 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를\n'
 '- 회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금\n'
 '- 을 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000648',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
