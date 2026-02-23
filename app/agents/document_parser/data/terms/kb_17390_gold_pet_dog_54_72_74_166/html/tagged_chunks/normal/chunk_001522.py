from langchain_core.documents import Document

chunk = Document(
    page_content=("있고, 유동식 섭취시<br>간헐적으로 흡인이 발생하고 부드러운 고형식 외에는 섭취가 불가 보</footer><br><p id='218' "
 "data-category='list'></p><br><p id='219' data-category='list' "
 "style='font-size:14px'>능한 상태 통약<br>5) 개구(입벌리기)장해는 턱관절의 이상으로 개구(입벌리기)운동 "
 '제한이<br>관<br>있는 상태를 말하며, 최대 개구(입벌리기)상태에서 위‧아래턱(상․하악)<br>의 가운데 앞니(중절치)간 거리를 '
 '기준으로 한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001522',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
