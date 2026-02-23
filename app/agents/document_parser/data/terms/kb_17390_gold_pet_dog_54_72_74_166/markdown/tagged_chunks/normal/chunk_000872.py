from langchain_core.documents import Document

chunk = Document(
    page_content=('‘손바닥 크기’라 함은 해당 환자의 손가락을 제외한 손바닥의 크기를 말하\n'
 '며, 12세 이상의 성인에서는 8×10㎝(1/2 크기는 40㎠, 1/4 크기는 20㎠),\n'
 '6~11세의 경우는 6×8㎝(1/2 크기는 24㎠, 1/4 크기는 12㎠), 6세 미만의 경\n'
 '우는 4×6㎝(1/2 크기는 12㎠, 1/4 크기는 6㎠)로 간주한다.- \n'
 '- 144 -# 6. 척추(등뼈)의장해| 가. 장해의 분류 |  |\n'
 '| --- | --- |\n'
 '| 장해의 분류 | 지급률 |\n'
 '| 1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000872',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
