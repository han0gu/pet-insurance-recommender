from langchain_core.documents import Document

chunk = Document(
    page_content=('받은 경우 이를 1회로 보아 제2항의 지급한도 내에서 지급\n'
 '합니다.\n'
 '\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여\n'
 '30일 이내에 발생한 질병은 보상하지 않습니다. 단,「반려\n'
 '동물 비용손해 관련 특별약관 일반조항」제15조(재가입) 제\n'
 '6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니\n'
 '다.152\uf000 제1항에도 불구하고 보장개시일로부터 그 날을 포함하여\n'
 '1년 이내에 발생한 슬관절탈구, 고관절탈구, 슬관절형성부\n'
 '전, 고관절형성부전(대퇴 골두 허혈성 괴사 포함) 또는 기'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000394',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
