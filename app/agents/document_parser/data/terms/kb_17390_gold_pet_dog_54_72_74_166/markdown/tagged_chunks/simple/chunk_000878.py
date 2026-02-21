from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6개월 이상 지난 후에 평가한다.\n'
 '- 5) 신경학적 검사상 나타난 저린감이나 방사통 등 신경자극증상의 원인으로\n'
 '- CT, MRI 등 영상검사에서 추간판탈출증이 확인된 경우를 추간판탈출증으\n'
 '- 로 진단하며, 수술 여부에 관계없이 운동장해 및 기형장해로 평가하지 보\n'
 '- 않는다. 통약\n'
 '- 6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.\n'
 '- 관\n'
 '- 가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 4개 이상의 척추체(척추뼈\n'
 '몸통)를 유합(아물어 붙음) 또는 고정한 상태'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000878',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
