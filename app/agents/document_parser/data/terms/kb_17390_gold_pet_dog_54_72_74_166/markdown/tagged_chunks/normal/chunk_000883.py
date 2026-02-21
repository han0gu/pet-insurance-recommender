from langchain_core.documents import Document

chunk = Document(
    page_content=('가 앞으로 휘어지는 증상), 척추후만증(척추가 뒤로 휘어지는 증상)\n'
 '또는 10° 이상의 척추측만증(척추가 옆으로 휘어지는 증상) 변형이\n'
 '있을 때\n'
 '나) 척추체(척추뼈 몸통) 한 개의 압박률이 40%이상인 경우 또는 한 운\n'
 '동단위 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추- \n'
 '체(척추뼈 몸통)의 압박률의 합이 60% 이상일 때11) 약간의 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.\n'
 '가) 1개 이상의 척추(등뼈)의 골절 또는 탈구로 경도(가벼운 정도)의 척'),
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
 'indexing': {'chunk_id': 'chunk_000883',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
