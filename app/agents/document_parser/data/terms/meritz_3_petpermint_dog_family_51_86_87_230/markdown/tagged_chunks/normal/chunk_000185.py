from langchain_core.documents import Document

chunk = Document(
    page_content=('한하는 방법을 말합니다.# 【일부보장 제외(부담보)】일반적인 경우보다 위험이 높은 반려동물이 가입하기 위\n'
 '한 방법의 하나로, 특정 질병 또는 특정 부위를 보장에\n'
 '서 제외하는 방법을 말합니다.# 【 보험금 삭감 】일반적인 경우보다 위험이 높은 반려동물이 가입하기 위\n'
 '한 방법의 하나로, 보험 가입 후 기간이 경과함에 따라\n'
 '위험의 크기 및 정도가 점차 감소하는 위험에 대해 적용\n'
 '하여 보험 가입 후 일정기간 내에 보험사고가 발생할 경\n'
 '우 미리 정해진 비율로 보험금을 감액하여 지급하는 방'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000185',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
