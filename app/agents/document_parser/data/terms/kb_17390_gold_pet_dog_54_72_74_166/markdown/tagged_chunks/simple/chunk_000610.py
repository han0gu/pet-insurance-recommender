from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 국가동물 미등록한 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를\n'
 '- 특\n'
 '- 회사에 제출하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금\n'
 '- 별\n'
 '- 을 지급합니다.\n'
 '- 약\n'
 '- 4. 사고증명서(진료비 영수증(치료비 세부내역 포함), 진료기록부(수의사가 작\n'
 '- 관\n'
 '- 성한 진료차트), MRI, CT, 방사선 촬영 등 영상검사를 하는 경우 해당 사진\n'
 '- (촬영 날짜 및 시간 필수)\n'
 '- - "이물제거(내시경)" 시행한 경우 : 이물제거(내시경) 처치가 명시된 진료비'),
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
 'indexing': {'chunk_id': 'chunk_000610',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
