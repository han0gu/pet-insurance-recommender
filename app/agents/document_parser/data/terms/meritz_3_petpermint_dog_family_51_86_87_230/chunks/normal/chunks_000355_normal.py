from langchain_core.documents import Document

chunk = Document(
    page_content=('타 이들과 유사한 질병 또는 상해에 대해서는 보험금을 지 급하지 않습니다. 단,「반려동물 비용손해 관련 특별약관 '
 '일반조항」제15조(재가입) 제6항에 따라 보험계약이 연장된 경우에는 적용하지 않습니다.\n'
 '【슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성 부전(대퇴 골두 허혈성 괴사 포함) 등 보장예시】\n'
 'Chart Type: bar\n'
 '보장개시일 | 슬관절탈구\n'
 'item_01 | 4.1 | 2.6\n'
 'item_02 | 0.4 | 4.1\n'
 '※ 설명 보장개시일로부터 1년이내 발생한 슬관절탈구 : 보험금 미지급'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000355',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
