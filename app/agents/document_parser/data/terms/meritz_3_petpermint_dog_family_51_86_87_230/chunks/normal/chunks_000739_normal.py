from langchain_core.documents import Document

chunk = Document(
    page_content=('3) 목\n'
 '손바닥 크기 1/2 이상의 추상(추한 모습)\n'
 '마. 손바닥 크기\n'
 '“손바닥 크기”라 함은 해당 환자의 손가락을 제외한 손 바닥의 크기를 말하며, 12세 이상의 성인에서는 8×10㎝ (1/2 크기는 '
 '40㎠, 1/4 크기는 20㎠), 6∼11세의 경우는 6×8㎝(1/2 크기는 24㎠, 1/4 크기는 12㎠), 6세 미만의 경우는 '
 '4×6㎝(1/2 크기는 12㎠, 1/4 크기는 6㎠)로 간 주한다.\n'
 '6. 척추(등뼈)의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 210},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000739',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
