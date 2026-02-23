from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나의 운동단위 내에서 여러 개의 척추체(척추뼈 몸통)에 압박골절이\n'
 '- 발생한 경우에는 각 척추체(척추뼈 몸통)의 압박률을 합산하고, 두\n'
 '- 개 이상의 운동단위에서 장해가 발생한 경우에는 그 중 가장 높은\n'
 '- 지급률을 적용한다.\n'
 '- 공\n'
 '- 3) 척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가 그 증상을 악화시킨\n'
 '- 통\n'
 '- 부분만큼, 즉 이 사고와의 관여도를 산정하여 평가한다.\n'
 '- 4) 추간판탈출증으로 인한 신경 장해는 수술 또는 시술(비수술적 치료) 후 사항\n'
 '- 6개월 이상 지난 후에 평가한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000877',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
