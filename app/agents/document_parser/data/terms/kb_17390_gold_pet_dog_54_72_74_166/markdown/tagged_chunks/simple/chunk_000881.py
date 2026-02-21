from langchain_core.documents import Document

chunk = Document(
    page_content=('에 뚜렷한 이상전위가 있는 상태- \n'
 '# 8) 약간의 운동장해머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추)를 제외한 척추체(척추뼈 몸통)에 골절 또는 탈구로 2개의 '
 '척추체(척추뼈 몸통)를 유합(아# 물어 붙음) 또는 고정한 상태9) 심한 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.가) '
 '척추(등뼈)의 골절 또는 탈구 등으로 35° 이상의 척추전만증(척추가 앞으로 휘어지는 증상), 척추후만증(척추가 뒤로 휘어지는 증상)\n'
 '또는 20° 이상의 척추측만증(척추가 옆으로 휘어지는 증상) 변형이\n'
 '있을 때'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000881',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
