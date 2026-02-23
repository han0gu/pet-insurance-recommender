from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>11) 약간의 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.<br>가) 1개 이상의 "
 '척추(등뼈)의 골절 또는 탈구로 경도(가벼운 정도)의 척<br>추전만증(척추가 앞으로 휘어지는 증상), 척추후만증(척추가 '
 '뒤로<br>휘어지는 증상) 또는 척추측만증(척추가 옆으로 휘어지는 증상) 변<br>형이 있을 때<br>나) 척추체(척추뼈 몸통) 한 개의 '
 '압박률이 20%이상인 경우 또는 한 운<br>동단위 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추</p><br><p'),
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
 'indexing': {'chunk_id': 'chunk_001563',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
