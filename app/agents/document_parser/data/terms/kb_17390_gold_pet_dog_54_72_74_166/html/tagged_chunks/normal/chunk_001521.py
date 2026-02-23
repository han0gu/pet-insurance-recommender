from langchain_core.documents import Document

chunk = Document(
    page_content=('저작(씹기)운동 제한<br>공<br>으로 부드러운 고형식(밥, 빵 등)만 섭취 가능한 경우<br>통<br>나) 위‧아래턱(상․하악)의 '
 '가운데 앞니(중절치)간 최대 개구(입벌리기)<br>운동이 2cm이하로 제한되는 경우 사항<br>다) 위‧아래턱(상․하악)의 '
 '부정교합(전방, 측방)이 1cm이상인 경우<br>라) 양측 각 1개 또는 편측 2개 이하의 치아만 교합되는 상태<br>마) '
 '연하기능검사(비디오 투시검사)상 연하장애가 있고, 유동식 섭취시<br>간헐적으로 흡인이 발생하고 부드러운 고형식 외에는 섭취가 불가'),
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
 'indexing': {'chunk_id': 'chunk_001521',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
