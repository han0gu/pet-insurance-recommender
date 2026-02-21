from langchain_core.documents import Document

chunk = Document(
    page_content=('. 청구서(회사 양식)<br>2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호<br>3. 국가동물 미등록한 경우에는 가입동물의 사진 '
 '2매(얼굴전면, 측면전신사진)를<br>회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금<br>을 '
 '지급합니다.<br>4. 사고증명서(진단서, 진료비계산서, 사망진단서, 장해진단서, 입원치료확인서,<br>의사처방전(처방조제비) '
 '등)<br>5'),
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
 'indexing': {'chunk_id': 'chunk_001293',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
