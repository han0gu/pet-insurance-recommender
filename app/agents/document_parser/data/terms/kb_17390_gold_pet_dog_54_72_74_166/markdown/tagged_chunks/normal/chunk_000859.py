from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 뚜렷한 개구(입벌리기)운동 제한 또는 뚜렷한 저작(씹기)운동 제한- \n'
 '으로 미음 또는 이에 준하는 정도의 음식물(죽 등)이외는 섭취하지\n'
 '못하는 경우\n'
 '나) 위‧아래턱(상․하악)의 가운데 앞니(중절치)간 최대 개구(입벌리기)\n'
 '운동이 1cm이하로 제한되는 경우\n'
 '다) 위‧아래턱(상․하악)의 부정교합(전방, 측방)이 1.5cm이상인 경우\n'
 '라) 1개 이하의 치아만 교합되는 상태\n'
 '마) 연하기능검사(비디오 투시검사)상 연하장애가 있고, 유동식 섭취\n'
 '시 흡인이 발생하고 연식 외에는 섭취가 불가능한 상태'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000859',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
